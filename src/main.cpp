#include "technology.h"
#include "chiplets.h"
#include <iostream>
#include <fstream>
#include "schnode.h"
#include "cluster.h"
#include "util.h"
#include "noc.h"
#include "layerengine.h"
#include "spatial_mapping/segmentation.h"
#include "debug.h"
#include "cost.h"

#include "nns/nns.h"

#include "ltreenode.h"

#include <cassert>
#include <time.h>

#include <string.h>
#include <math.h>
#include <random>

#include <type_traits>
#include <stdexcept>
#include <map>
#include <unordered_set>
#include <fstream>
#include <sstream>

#include <thread>
#include <mutex>

#include "json/json.h"
#include "schnode.h"

#define KB *1024

using namespace std;

struct WholeSch{
	LTreeNode* tree;
	SchNode* sch;
	WholeSch(): tree(nullptr), sch(nullptr){}
	WholeSch(LTreeNode* _tree, SchNode* _sch): tree(_tree), sch(_sch){}
	operator bool() const{
		return tree;
	}
	WholeSch copy() const{
		if(!tree) return {nullptr, nullptr};
		LTreeNode* new_tree = tree->copy();
		SchNode* new_sch = sch->copy();
		return WholeSch(new_tree, new_sch);
	}
	void del(){
		if(tree){
			delete tree;
			delete sch;
		}
	}
	void min(WholeSch& w_sch){
		if(!tree){
			tree = w_sch.tree;
			sch = w_sch.sch;
		}else if(sch->get_cost().cost() > w_sch.sch->get_cost().cost()){
			del();
			tree = w_sch.tree;
			sch = w_sch.sch;
		}else{
			w_sch.del();
		}
		w_sch.tree = nullptr;
		w_sch.sch = nullptr;
	}
	/*
	~WholeSch(){
		del();
	}
	*/
};

size_t find(const LTreeNode::node_vec& vec, LTreeNode* node){
	for(size_t i=0; i<vec.size(); ++i){
		if(vec[i] == node) return i;
	}
	assert(false);
	return vec.size();
}

void halv_bat(LTreeNode* node){
	if(!node->children.empty() && node->children.front()->num_batch == node->num_batch){
		for(auto x:node->children){
			halv_bat(x);
		}
	}
	node->num_batch /= 2;
}

void flat_bat(LTreeNode* node, len_t n_batch = 1){
	if(node->num_batch <= n_batch) return;
	for(auto x:node->children){
		flat_bat(x, n_batch);
	}
	node->num_batch = n_batch;
}

class SAEngine{
public:
	static int nrounds;
private:
	static constexpr int NUM_OP = 7;
	int cur_round;
	uint64_t num_tries;
	uint64_t cur_tries;
	std::mt19937 generator;
	ostringstream strStream;
	ostream& out;
	// Generate int between [0, to)
	int randInt(int to){
		return std::uniform_int_distribution<int>(0, to-1)(generator);
	}
	bool withProb(double prob){
		return std::uniform_real_distribution<double>(0.0, 1.0)(generator) < prob;
	}
public:
	SAEngine(std::uint32_t seed, bool directCout = false):generator(seed), out(directCout?cout:strStream){
		strStream.precision(4);
	}

	void flushBuf(){
		cout << strStream.str() << flush;
		strStream.clear();
		strStream.str("");
	}

	// sa_type: 0 -> arbitrary. 1 -> only s under top t. 2 -> only t under top t.
	LTreeNode* sa_change(LTreeNode* root, bool* valid_op, lid_t max_depth=0, int sa_type=0, int* op_type=nullptr){
		root = root->copy();
		lid_t lnum = root->layers().count();
		if(max_depth == 0) max_depth = lnum;
		if(root->height > max_depth){
			throw std::invalid_argument("The root of SA is deeper than max_depth!");
		}

		int prob[NUM_OP]={10,10,20,20,20,20,40};
		for(int i=1;i<NUM_OP;++i){
			prob[i] += prob[i-1];
		}
		//bool x[NUM_OP+1];
		//memset(x,0,NUM_OP);
		int t=NUM_OP;
		uint64_t print_tries = 30;
		cur_tries = 0;
		bool ok;
		// For convenience, we research the whole seg now.
		assert(root->get_type() == LTreeNode::NodeType::T);
		do{
			//x[t] = true;
			lid_t l = randInt(lnum);
			LTreeNode* lnode=root;
			lid_t depth = 0;
			while(lnode->t != LTreeNode::NodeType::L){
				for(auto child: lnode->children){
					if(child->layers().contains(l)){
						lnode = child;
						++depth;
						break;
					}
				}
			}
			assert(depth > 0);
			ok=false;
			do{
				int pr = randInt(prob[NUM_OP-1]);
				t=0;
				while(pr >= prob[t]) ++t;
			}while(!valid_op[t]);
			//if(x[t]) continue;
			switch (t) {
			case 0:{
				// Change with front.
				LTreeNode* front = lnode->parent;
				LTreeNode* c = lnode;
				while (front && front->children.front() == c) {
					c = front;
					front = front->parent;
				}
				if(!front) break;
				LTreeNode* lcl = c;
				size_t k = find(front->children, c);
				//cout << k << " is " << endl;
				c = front->children[k-1];
				LTreeNode* lcc = c;
				while (!c->children.empty()) {
					c = c->children.back();
				}
				lid_t x = c->layer_set.first();
				if(network->getNode(l).getPrevs().contains(x)) break;
				// Found valid front, change!
				front->stage.clear();
				// Reset path to lcl/lcc.
				while(lcl!=lnode){
					lcl->layer_set.reset(l);
					lcl->layer_set.set(x);
					lcl->stage.clear();
					for (auto z:lcl->children) {
						if(z->layer_set.contains(l)){
							lcl = z;
							break;
						}
					}
				}
				while(lcc!=c){
					lcc->layer_set.reset(x);
					lcc->layer_set.set(l);
					lcc->stage.clear();
					lcc = lcc->children.back();
				}
				size_t i = find(c->parent->children, c);
				size_t j = find(lnode->parent->children, lnode);
				c->parent->children[i] = lnode;
				lnode->parent->children[j] = c;
				swap(lnode->parent,c->parent);
				swap(lnode->num_batch,c->num_batch);
				//std::cout << (int)x << ' ' << (int)l <<std::endl;
				// Now we'll reset the whole seg.
				if(front == root){
					front->children[k]->isNewNode = true;
					front->children[k-1]->isNewNode = true;
				}else{
					while(front->parent->parent) front = front->parent;
					front->isNewNode = true;
				}
				ok=true;
			}break;
			case 1:{
				// Change with back.
				LTreeNode* back = lnode->parent;
				LTreeNode* c = lnode;
				while (back && back->children.back() == c) {
					c = back;
					back = back->parent;
				}
				if(!back) break;
				LTreeNode* lcl = c;
				size_t k = find(back->children, c);
				//cout << k << " is " << endl;
				c = back->children[k+1];
				LTreeNode* lcc = c;
				while (!c->children.empty()) {
					c = c->children.front();
				}
				lid_t x = c->layer_set.first();
				if(network->getNode(x).getPrevs().contains(l)) break;
				// Found valid back, change!
				back->stage.clear();
				// Reset path to lcl/lcc.
				while(lcl!=lnode){
					lcl->layer_set.reset(l);
					lcl->layer_set.set(x);
					lcl->stage.clear();
					for (auto z:lcl->children) {
						if(z->layer_set.contains(l)){
							lcl = z;
							break;
						}
					}
				}
				while(lcc!=c){
					lcc->layer_set.reset(x);
					lcc->layer_set.set(l);
					lcc->stage.clear();
					lcc = lcc->children.front();
				}
				size_t i = find(c->parent->children, c);
				size_t j = find(lnode->parent->children, lnode);
				c->parent->children[i] = lnode;
				lnode->parent->children[j] = c;
				swap(lnode->parent,c->parent);
				swap(lnode->num_batch,c->num_batch);
				//std::cout << (int)x << ' ' << (int)l <<std::endl;
				// Now we'll reset the whole seg.
				if(back == root){
					back->children[k]->isNewNode = true;
					back->children[k+1]->isNewNode = true;
				}else{
					while(back->parent->parent) back = back->parent;
					back->isNewNode = true;
				}
				ok=true;
			}break;
			case 2:{
				// Delete parent, merge to grandma.
				LTreeNode* par = lnode->parent;
				if(par == nullptr) break;
				LTreeNode* grandma = par->parent;
				if(grandma == nullptr) break;
				size_t i = find(grandma->children, par);
				grandma->children.erase(grandma->children.begin()+i);
				grandma->children.insert(grandma->children.begin()+i, par->children.begin(), par->children.end());
				grandma->stage.clear();
				for(auto x : par->children){
					x->parent = grandma;
					x->t = LTreeNode::NodeType::L;
					x->num_batch = par->num_batch;
				}
				// Now we'll reset the whole seg.
				if(grandma == root){
					for(auto x : par->children){
						x->isNewNode = true;
					}
				}else{
					while(grandma->parent->parent) grandma = grandma->parent;
					grandma->isNewNode = true;
				}
				par->children.clear();
				delete par;
				ok=true;
			}break;
			case 3:{
				// Add new parent, select a range of childs.
				LTreeNode* par = lnode->parent;
				if(par == nullptr || par->children.size() <= 2) break;
				size_t i = find(par->children, lnode);
				size_t x= par->children.size()-1;
				size_t j;
				do{
					j=randInt(x);
					if(j>=i)++j;
				}while(i*j==0&&i+j==x);
				x = MIN(i,j);
				j=MAX(i,j)+1;
				i=x;
				// Check depth limit
				if(par->height + depth > max_depth){
					bool p = false;
					for(size_t x=i;x<j;++x){
						if(par->children[x]->height + depth >= max_depth){
							p=true;
							break;
						}
					}
					if(p) break;
				}
				par->stage.clear();
				LTreeNode* new_par;
				bool T_under_T = false;
				if((par->parent == nullptr) && (par->t == LTreeNode::NodeType::T)){
					switch(sa_type){
						case 0: T_under_T = withProb(0.5); break;
						case 1: break;
						case 2: T_under_T = true; break;
						default: assert(false);
					}
				}
				new_par=new LTreeNode(Bitset(),lnode->num_batch,nullptr, T_under_T ? LTreeNode::NodeType::T : LTreeNode::NodeType::L);
				new_par->children.insert(new_par->children.begin(), par->children.begin()+i, par->children.begin()+j);
				new_par->parent = par;
				par->children.erase(par->children.begin()+i, par->children.begin()+j);
				par->children.insert(par->children.begin()+i, new_par);
				for(auto x:new_par->children){
					x->parent = new_par;
					if(T_under_T) x->t = LTreeNode::NodeType::L;
					else if(par->t == LTreeNode::NodeType::T) flat_bat(x);
				}
				// Now we'll reset the whole seg.
				while(new_par->parent->parent) new_par = new_par->parent;
				new_par->isNewNode = true;
				ok=true;
			}break;
			case 4:{
				// Put batch down
				LTreeNode* cur = lnode->parent;
				int d = randInt(depth);
				while(d-->0){
					cur = cur->parent;
				}
				if(cur->children.front()->num_batch == cur->num_batch){
					break;
				}
				for(auto x:cur->children){
					x->num_batch *= 2;
				}
				// Now we'll reset the whole seg.
				if(cur == root){
					cur->isNewNode = true;
				}else{
					while(cur->parent->parent) cur = cur->parent;
					cur->isNewNode = true;
				}
				ok=true;
			}break;
			case 5:{
				// Put batch up
				LTreeNode* cur = lnode->parent;
				int d = randInt(depth);
				while(d-->0){
					cur = cur->parent;
				}
				if(cur->children.front()->num_batch == 1){
					break;
				}
				for(auto x:cur->children){
					halv_bat(x);
				}
				// Now we'll reset the whole seg.
				if(cur == root){
					cur->isNewNode = true;
				}else{
					while(cur->parent->parent) cur = cur->parent;
					cur->isNewNode = true;
				}
				ok=true;
			}break;
			case 6:{
				// Put lnode into the cut before/after it (or out of its parent).
				LTreeNode* par = lnode->parent;
				if(par->children.size() <= 2) break;
				size_t node_pos = find(par->children, lnode);
				bool put_before = (randInt(2) == 0);
				if(put_before?(node_pos > 0):(node_pos < par->children.size()-1)){
					size_t next_pos = node_pos + (put_before?-1:1);
					LTreeNode* cut = par->children[next_pos];
					if(cut->t == LTreeNode::NodeType::L) break;
					// Put lnode under cut.
					par->stage.clear();
					cut->stage.clear();
					par->children.erase(par->children.begin()+node_pos);
					if(put_before){
						cut->children.push_back(lnode);
					}else{
						cut->children.insert(cut->children.begin(), lnode);
					}
					lnode->parent = cut;

					lnode->num_batch = cut->get_bgrp_size();
					cut->layer_set.set(l);
					// Now we'll reset the whole seg.
					assert(cut != root);
					while(cut->parent->parent) cut = cut->parent;
					cut->isNewNode = true;
				}else{
					LTreeNode* grandma = par->parent;
					if(grandma == nullptr) break;
					// Put lnode under par->parent.
					size_t par_pos = find(grandma->children, par);
					size_t insert_pos = par_pos + (put_before?0:1);
					par->stage.clear();
					grandma->stage.clear();
					par->children.erase(par->children.begin()+node_pos);
					grandma->children.insert(grandma->children.begin()+insert_pos, lnode);
					lnode->parent = grandma;

					lnode->num_batch = grandma->get_bgrp_size();
					par->layer_set.reset(l);
					// Now we'll reset the whole seg.
					if(grandma == root){
						lnode->isNewNode = true;
						par->isNewNode = true;
					}else{
						while(grandma->parent->parent) grandma = grandma->parent;
						grandma->isNewNode = true;
					}
				}
				ok=true;
			}break;
			default:
				break;
			}
			++cur_tries;
			++num_tries;
			if(cur_tries >= print_tries){
				out << "[Warning] After " << cur_tries << " tries." << endl;
				print_tries *= 2;
			}
		}while(!ok);
		//std::cout << t << std::endl;
		if(op_type) *op_type = t;
		root->init_root();
		return root;
	}

	bool sa_accept(cost_t cur_cost, cost_t new_cost, int round){
		if(new_cost <= cur_cost) return true;
		/*
		 * T(x) = a+c/(b+x)
		 * a + c/b = 0.1
		 * a + c/(b+0.5) = 0.01
		 * a + c/(b+1) = 0
		 * a = -1 / 80
		 * b = 0.01*0.5/(0.1*0.5-0.01) = 0.125
		 * c = 9 / 640
		 * T(x) = 1/10 * (1-x)/(1+8x)
		 */
		double x = round;
		x /= nrounds;
		// Since only 1/100 are good, multiply T by 0.7:
		double T = 0.07 * (1-x)/(1+8*x);
		double prob = exp(-((new_cost - cur_cost)/cur_cost)/T);
		//cout << x << ' ' << T << ' ' << prob << std::endl;
		return withProb(prob);
	}

	/*
	static mutex m;
	void ping_func(volatile bool& stop){
		while(true){
			for(int i=0;i<120;++i){
				this_thread::sleep_for(chrono::milliseconds(500));
				if(stop) return;
			}
			unique_lock<mutex> l(m);
			cout << "[Ping] " << cur_round << ' ' << cur_tries << ' ' << num_tries << endl;
			l.unlock();
		}
	}
	*/

	void SA_search(WholeSch& w_sch, const Cluster& c, lid_t max_depth=0, int sa_type=0){
		time_t start_time = time(nullptr);
		int nvalid = 0, naccept = 0;
		LTreeNode*& min_node = w_sch.tree;
		SchNode*& min_res = w_sch.sch;
		LTreeNode* cur_node = w_sch.tree;
		SchNode* cur_res = w_sch.sch;
		int op_type;
		map<int, int> accept_num;
		map<int, int> valid_num;
		for(int i=0;i<NUM_OP;++i){
			accept_num[i] = 0;
			valid_num[i] = 0;
		}
		bool using_best = false;

		bool valid_op[NUM_OP];
		for(int i=0;i<NUM_OP;++i) valid_op[i] = true;
		if(network->is_chain()) valid_op[0] = valid_op[1] = false;
		if(cur_node->get_tot_batch() == 1) valid_op[4] = valid_op[5] = false;
		num_tries = 0;
		cur_tries = 0;
		int print_intv = nrounds/30;
		cur_round=0;
		// bool stop_ping = false;
		// std::thread ping(ping_func, ref(stop_ping));
		for(;cur_round<nrounds;++cur_round){
			if((cur_round+1)%print_intv == 0){
				//unique_lock<mutex> l(m);
				out << cur_round << ' ' << cur_res->get_cost().cost() << ' ' << (num_tries * 1.0) /print_intv << endl;
				//l.unlock();
				num_tries = 0;
			}
			if(cur_round >= 0.90*nrounds && !using_best){
				using_best = true;
				if(cur_node != min_node){
					//unique_lock<mutex> l(m);
					out << "Switch to best solution." << endl;
					//l.unlock();
					delete cur_node;
					delete cur_res;
					cur_node = min_node;
					cur_res = min_res;
				}
			}
			LTreeNode* new_tree = sa_change(cur_node, valid_op, max_depth, sa_type, &op_type);
			SchNode* new_res;
			if(new_tree->isNew()){
				new_res = Cut::newNode(new_tree, c, nullptr);
			}else{
				new_res = cur_res->copy();
				assert(new_tree->isModified());
				new_res->searchInc(new_tree,false);
			}
			if(!new_res->is_valid()){
				delete new_tree;
				delete new_res;
				continue;
			}
			new_tree->confirm();
			++nvalid;
			++valid_num[op_type];
			cost_t new_cost = new_res->get_cost().cost();
			if(new_cost < min_res->get_cost().cost()){
				if(cur_node != min_node){
					delete min_node;
					delete min_res;
				}
				min_node = new_tree;
				min_res = new_res;
			}

			if(sa_accept(cur_res->get_cost().cost(), new_cost, cur_round)){
				//cout << "accept!" << endl;
				if(cur_node != min_node){
					delete cur_node;
					delete cur_res;
				}
				cur_node = new_tree;
				cur_res = new_res;
				++naccept;
				++accept_num[op_type];
			}else{
				if(new_tree != min_node){
					delete new_tree;
					delete new_res;
				}
			}
		}
		if(cur_node != min_node){
			delete cur_node;
			delete cur_res;
		}
		time_t end_time = time(nullptr);
		//stop_ping = true;
		//ping.join();
		out << "Elapsed: " << end_time - start_time << "s ";
		out << "Valid: " << nvalid << " (" << (nvalid*100.0)/nrounds << "%) ";
		out << "Accept: " << naccept << " (" << (naccept*100.0)/nrounds << "%)" << endl;
		out << "Per OP: ";
		for(int i=0;i<NUM_OP;++i){
			if(i>0) out << ", ";
			out << accept_num[i] << '/' << valid_num[i];
		}
		out<<endl;
	}
};

int SAEngine::nrounds;

void LP_search(lid_t num_layer, len_t tot_batch, Cluster& c, WholeSch& w_sch, bool has_S, bool has_T){
	if(!has_S && !has_T){
		throw std::invalid_argument("Either has_S or has_T must be true.");
	}
	LTreeNode*& tree_node = w_sch.tree;
	SchNode*& sch_res = w_sch.sch;
	SchNode::SchCost LP_cost;
	LTreeNode** LP_DP = new LTreeNode*[num_layer];
	SchNode* tmp = nullptr;
	SchNode* last = nullptr;
	LTreeNode* root_Node;
	for(int i=0;i<num_layer;++i) {
		LP_DP[i]=nullptr;
		LP_cost.energy = energy_inf;
		// cout << "\tStart " << network->getNode(i).name() << endl;
		for(int j=-1;j<i;++j){
			if(j != -1 && LP_DP[j] == nullptr) continue;
			for(len_t l_bat=1;l_bat<=4&&l_bat<=tot_batch;l_bat*=2){
				for(int last_T = has_S?0:1; last_T < (has_T?2:1); ++last_T){
					if(j==i-1 && has_S && has_T && last_T > 0) break;
					if(j==-1){
						// Create top T-cut.
						root_Node = new LTreeNode(Bitset(), tot_batch, nullptr, LTreeNode::NodeType::T);
					}else {
						// Use top T-cut from DP.
						root_Node = LP_DP[j]->copy();
						root_Node->reset_lset();
					}
					if(j == i-1){
						// New LNode: last layer.
						(void) new LTreeNode(i, tot_batch, root_Node);
					}else{
						LTreeNode* cur_Cut = new LTreeNode(Bitset(), tot_batch, root_Node, (last_T == 0) ? LTreeNode::NodeType::S : LTreeNode::NodeType::T);
						for(int k=j+1;k<=i;++k){
							(void) new LTreeNode(k, l_bat, cur_Cut);
						}
					}
					root_Node->init_root();
					// TODO: change to inc. search
					tmp = Cut::newNode(root_Node, c, nullptr);
					if(!tmp->is_valid()){
						delete tmp;
						delete root_Node;
						continue;
					}
					root_Node->confirm();
					if(tmp->get_cost().cost() >= LP_cost.cost()){
						delete tmp;
						delete root_Node;
						continue;
					}
					if(LP_DP[i]) delete LP_DP[i];
					LP_DP[i] = root_Node;
					LP_cost = tmp->get_cost();
					if(last) delete last;
					last = tmp;
				}
			}
		}
	}
	tree_node = LP_DP[num_layer-1];
	if(tree_node == nullptr){
		std::cout << "Warning: no scheme found in LP_search!" << std::endl;
		sch_res = nullptr;
		if(last) delete last;
	}else{
		sch_res = last;
	}
	for(int i=0; i<num_layer-1; ++i) if(LP_DP[i]) delete LP_DP[i];
	delete[] LP_DP;
	return;
}

// buffer(): the SRAM access-energy table lives in chiplets.cpp (H3).
double wafer_util(double area) {
	if (	 area <= 25000000) {
		return 0.865;
	}
	else if (area <= 50000000) {
		return 0.859;
	}
	else if (area <= 100000000) {
		return 0.849;
	}
	else if (area <= 200000000) {
		return 0.815;
	}
	else if (area <= 300000000) {
		return 0.806;
	}
	else if (area <= 400000000) {
		return 0.792;
	}
	else if (area <= 500000000) {
		return 0.778;
	}
	else if (area <= 600000000) {
		return 0.764;
	}
	else if (area <= 700000000) {
		return 0.743;
	}
	else if (area <= 800000000) {
		return 0.743;
	}
}
int main(int argc, char** argv){
	if (argc > 1 && std::string(argv[1]) == "--dump-technology") {  // chess/main: print the built-in Technology of a stdin line as a file
		string t, pk, io, dd; int skip; string s_;
		cin >> t >> skip >> skip >> skip >> skip >> skip >> skip >> skip >> skip >> skip >> skip >> pk >> io >> skip >> dd;
		Technology::builtin(t, pk, io, dd).dump(std::cout);
		return 0;
	}
	const char* seed_env = getenv("GEMINI_SEED");  // chess/main: a fixed seed makes two runs comparable
	unsigned int seed = seed_env ? (unsigned int)atol(seed_env) : time(NULL);
	srand(seed);
	cout << "Seed: " << seed << " ";
	std::cout.precision(4);

	// Get input parameters.
	int mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, _NoP_bw, _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops;
	string tech, package_type, IO_type, DDR_type, net_tok;
	cin >> tech >> mm >> net_tok >> xx >> yy >> ss >> bb >> rr >> ff >> xcut >> ycut >> package_type >> IO_type >> _NoP_bw >> DDR_type >> _DRAM_bw >> _NoC_bw >> _mac_dim >> _ul3 >> total_tops;
	// chess/main (E8): the network token is a built-in network's number, or `@<file>` for a network read from a file (src/nns/network_file.cpp; nn is then -1)
	string net_file;
	if (net_tok.rfind("@", 0) == 0) {
		net_file = net_tok.substr(1);
		nn = -1;
	} else {
		nn = stoi(net_tok);
	}
	// chess H3: an optional 21st token `@<chiplet map>` gives every chiplet its core type and technology (see include/chiplets.h)
	string chiplet_path;
	{
		string tok;
		if (cin >> tok) {
			if (tok.rfind("@", 0) == 0) chiplet_path = tok.substr(1);
			else throw runtime_error("unexpected token after total_tops: " + tok);
		}
	}
	// assert(total_tops == 2*xx*yy*_mac_dim*_mac_dim);
	//mm means core_microarch: 0 means PolarCore (NVDLA-STYLE), which is recommenmanded; 1 means Eyeriss Core.
	// chess H4: 2 a weight-stationary systolic array, 3 a vector/SFU engine (no contractions).
	//nn means network choice
	//xx & yy means the number of cores in the x and y axis, respectively. Default noc topology is mesh.
	//ss means stride, whose definition can be found in ISCA2023 SET, HPCA2024GEMINI, ASPLOS2019TANGRAM.
	//bb means batch size (2^n is recommanded).
	//rr means exploration rounds number, which will be multiplied with a coefficient.
	//ff means optimization goal, concrete formula can be found in this main.cpp.
	//xcut & ycut means the chiplet partition granularity at the x and y dimension.
	//the remaining few parameters represent the architecture parameters as their names.
	/********************* INPUT *********************/
	bw_t NoP_bw = _NoP_bw;
	NoC::DRAM_bw = _DRAM_bw/1024/4;     
	SchNode::DRAM_bw = _DRAM_bw/1024/4;   
	double DRAM_bw_each = _DRAM_bw / 1024 / 4;
	NoC::NoC_bw = _NoC_bw;     
	//core & chiplet num
	Cluster::xlen = xx;     
	Cluster::ylen = yy;     
	NoC::x_cut = xcut;   
	NoC::y_cut = ycut;
	NoC::soc = xcut * ycut == 1;
	std::uint16_t mac_dim = _mac_dim;  
	vol_t ul3_ = _ul3 KB;            

	NoC::NoP_bw = _NoP_bw;
	NoC::x_step = Cluster::xlen / NoC::x_cut;
	NoC::y_step = Cluster::ylen / NoC::y_cut;
	if (xx % ss != 0 || xx % NoC::x_cut != 0 || yy % NoC::y_cut != 0) {
		throw runtime_error("Chiplet partition is invalid.");
	}
    
	// chess/main (T3): every technology number comes from one Technology, built in or loaded from "@<file>".
	Technology T = tech.rfind("@", 0) == 0 ? Technology::load(tech.substr(1)) : Technology::builtin(tech, package_type, IO_type, DDR_type);
	tech = T.node;
	SchNode::frequency = T.frequency_ghz;
	//DENSITY (package level: the IO die and the NoP PHY take the base technology; a chiplet type's node terms are its own)
	density_t NoP_den = T.nop_den;
	density_t DDR_PHY_den = T.ddr_phy_den;
	density_t DDR_ctrl_den = T.ddr_ctrl_den;
	double ddr_cost = T.ddr_cost;
	NoC::DRAM_acc_cost = T.dram_acc_cost;
	NoC::seperate_IO = true;
	ofm_ubuf_vol = 10 KB;

	// chess H3: one core type, mapper and technology per chiplet type; upstream's package when no map is given
	ChipletMap chiplets = chiplet_path.empty() ? ChipletMap::homogeneous(mm, mac_dim, ul3_, T) : ChipletMap::load(chiplet_path, T);
	SchNode::chiplets = &chiplets;
	NoC::chiplets = &chiplets;
	SchNode::frequency = chiplets.frequency_ghz();  // the reference clock: the interconnect's, and the unit of the result's time (the base technology's by default)
	// NoC::NoP_hop_cost = 2 * 8; //DSE->3.3*8 ; Simba->2*8
	NoC::NoC_hop_cost = chiplets.noc_hop_cost();

	double tops = chiplets.tops();
	//NUMBER
	double core_num = Cluster::xlen * Cluster::ylen;
	double compute_die_num = NoC::x_cut * NoC::y_cut;
	double NoP_side_num;//how many sides of 
	double IO_die_num = 0;
	int serdes_number = 0;//calculate the number of serdes for all noc nodes (assuming each side of a node has a serdes)
	if (compute_die_num == 1) {
		NoP_side_num = 0;
		IO_die_num = 0;
	}
	else if (compute_die_num == 2) {
		NoP_side_num = 2;
		IO_die_num = 4;
		if(T.serdes){
			serdes_number = Cluster::ylen*2+2*Cluster::ylen*2;
		}
	}
	else {
		NoP_side_num = 4;
		IO_die_num = 4;
		if(T.serdes){
			serdes_number = Cluster::ylen*2+2*Cluster::ylen*NoC::x_cut+2*Cluster::xlen*NoC::y_cut;
		}
	}

	NoC::serdes = T.serdes;
	SchNode::serdes = T.serdes;
	if (T.serdes) {
		SchNode::serdes_power = T.serdes_power_mw;
		SchNode::serdes_lane_total = NoP_bw*8/static_cast<bw_t>(T.serdes_lane_gbps)*serdes_number;
	} else {
		NoC::NoP_hop_cost = T.nop_hop_cost;
	}
	//COST
	double control_unit_prop = T.control_unit_prop;
	double DFT_prop = T.dft_prop;

	//*********************DIE AREA*********************
	chiplets.compute_areas(T, NoC::NoC_bw, NoP_bw);
	double PCIe_area = 3000000 * DRAM_bw_each * 4 / 128;
	double IO_die_area = DRAM_bw_each * 4 * (DDR_PHY_den+DDR_ctrl_den)+ PCIe_area +(compute_die_num==1?0:(NoP_den * NoC::NoP_bw *Cluster::ylen*2));//neglect other IOs
	IO_die_area *= (control_unit_prop * DFT_prop);
	IO_die_area /= 1000000;
	double compute_die_area = chiplets.compute_die_area_mm2();  // the type's area, or the mean over chiplets of a heterogeneous package
	double total_die_area = chiplets.total_compute_die_area_mm2() + IO_die_area;

	// Check if areas are too large.
	if (compute_die_num == 1) {
		if (total_die_area > T.max_die_area_mm2) {
			cout << "`total_die_area` too large." << endl;
			return 1;
		}
	}else{
		if(IO_die_area>T.max_die_area_mm2 || chiplets.max_compute_die_area_mm2()>T.max_die_area_mm2) {
			cout << "`IO_die_area` or `compute_die_area` too large." << endl;
			return 1;
		}
	}
	//*********************DIE AREA*********************
	//*********************COST CALC*********************
	double cost_overall = 0;
	double cost_system_package = 0;
	double cost_soc = 0;
	double cost_chip = 0;
	double cost_package = 0;
	{
		// one chip configuration per chiplet type (its node, area and count), the IO dies when the package has several compute dies
		std::vector<ChipConfig> configs;
		for (const ChipletType& ct : chiplets.types) {
			configs.push_back({ct.tech.node, "Chip", package_type, ct.compute_die_area_mm2, ct.num_chiplets});
		}
		if (compute_die_num != 1) {
			configs.push_back({tech, "IO", package_type, IO_die_area/4, 4});
		}
		auto [total_cost, total_cost_chip, total_cost_package] = calculate_cost(configs);
		cost_chip = total_cost_chip;
		cost_package = total_cost_package;
		cost_system_package = total_cost;
		if (compute_die_num == 1) cost_soc = cost_system_package;
		ddr_cost = DRAM_bw_each * 4 * ddr_cost;
		cost_overall = cost_system_package + ddr_cost;
	}
	//*********************COST CALC*********************

	// TOPS
	

	// 2 GB/TOPS
	if (NoC::DRAM_bw == 0) {
		NoC::DRAM_bw = 0.75 * (tops / 4);
		SchNode::DRAM_bw = 0.75 * (tops / 4);
		NoC::NoC_bw = 4;//NoC::DRAM_bw / 4;
	}
	NoC::interleave = false;

	StdLayerEngine engine(&chiplets);
	FastLayerEngine engine_fast(&chiplets);
	SchNode::layerMapper = &engine;
	SchNode::layerMapper_fast = &engine_fast;
	Light_placement::fast_buffer = &engine_fast;

	std::string net_name;
	if (!net_file.empty()) {
		static Network file_network = load_network(net_file, net_name);
		network = &file_network;
	} else switch (nn) {
	case 0:
		network = &darknet19;
		net_name="darknet19";
		break;
	case 1:
		network = &vgg19;
		net_name="vgg";
		break;
	case 2:
		network = &resnet50;
		net_name="resnet";
		break;
	case 3:
		network = &googlenet;
		net_name="goog";
		break;
	case 4:
		network = &resnet101;
		net_name="resnet101";
		break;
	case 5:
		network = &densenet;
		net_name="densenet";
		break;
	case 6:
		network = &inception_resnet_v1;
		net_name="ires";
		break;
	case 7:
		network = &gnmt;
		net_name="gnmt";
		break;
	case 8:
		network = &lstm;
		net_name="lstm";
		break;
	case 9:
		network = &zfnet;
		net_name="zfnet";
		break;
	case 10:
		network = &transformer;
		net_name="trans";
		break;
	case 11:
		network = &transformer_cell;
		net_name="trans_cell";
		break;
	case 12:
		network = &PNASNet;
		net_name="pnas";
		break;
	case 13:
		network = &resnext50;
		net_name="resnext50";
		break;
	case 14:
		network = &resnet152;
		net_name="resnet152";
		break;
	case 15:
		network = &BERT_block;
		net_name="bert";
		break;
	case 16:
		network = &GPT2_prefill_block;
		net_name="gpt_prefill";
		break;
	case 17:
		network = &GPT2_decode_block;
		net_name="gpt_decode";
		break;
	// TODO: Support more DNNs.
	default:
		throw runtime_error("Model not supported.");
	}

	
	len_t tot_batch = bb;
	SchNode::tot_batch = tot_batch;

	Cluster::stride = ss;
	Cluster::min_util = 0.75;

	//SchNode::tot_batch = 64;
	lid_t num_layer = network->len();
	SAEngine::nrounds = rr;
	network->set_utime(chiplets);
	
	Cluster c(0,Cluster::xlen * Cluster::ylen);

	
	
	//NoP link caculation
	if (compute_die_num != 1) {
		for (mlen_t i = 0; i < Cluster::ylen; i++) {
			NoC::NoP_links.insert((- 1* Cluster::ylen + i) * 4);
			NoC::NoP_links.insert((0 * Cluster::ylen + i) * 4+2);
			NoC::NoP_links.insert(((Cluster::xlen-1)* Cluster::ylen + i) * 4);
			NoC::NoP_links.insert((Cluster::xlen * Cluster::ylen + i) * 4 + 2);
			for (mlen_t j = Cluster::xlen / NoC::x_cut; j < Cluster::xlen;) {
				NoC::NoP_links.insert(((j - 1) * Cluster::ylen + i) * 4);
				NoC::NoP_links.insert((j * Cluster::ylen + i) * 4 + 2);
				j += Cluster::xlen / NoC::x_cut;

			}
		}

		for (mlen_t i = 0; i < Cluster::xlen; i++) {
			for (mlen_t j = Cluster::ylen / NoC::y_cut; j < Cluster::ylen;) {
				NoC::NoP_links.insert((i * Cluster::ylen + j - 1) * 4 + 3);
				NoC::NoP_links.insert((i * Cluster::ylen + j) * 4 + 1);
				j += Cluster::ylen / NoC::y_cut;
			}
		}
	}


	NoC::DRAM_num = 4;
	NoC::DRAM_router_num = Cluster::ylen / 2;
	NoC::dram_list.resize(NoC::DRAM_num);
	for (mlen_t y = 0; y < NoC::DRAM_num; ++y) {
		NoC::dram_list[y].resize(NoC::DRAM_router_num);
	}

	for (mlen_t y = 0; y < Cluster::ylen; ++y) {
		if (y < NoC::DRAM_router_num) {
			NoC::dram_list[0][y] = {-1, y};
			NoC::dram_list[2][y] = {static_cast<mlen_t>(Cluster::xlen), y};
		}
		else if(y>= Cluster::ylen - NoC::DRAM_router_num){
			NoC::dram_list[1][y - Cluster::ylen / 2] = { -1, y };
			NoC::dram_list[3][y - Cluster::ylen / 2] = { static_cast<mlen_t>(Cluster::xlen), y };
		}
	}
	
	NoC::dram_list_base.resize(2*Cluster::ylen);
	for (mlen_t y = 0; y < Cluster::ylen; ++y) {
		NoC::dram_list_base[y] = { -1, y};
		NoC::dram_list_base[Cluster::ylen + y] = { static_cast<mlen_t>(Cluster::xlen), y };
	}

	switch (ff){
		case 1:
			// This is the default cost_func.
			cost_func = [](energy_t e, cycle_t t){return e*t;};
			break;
		case 0:
			cost_func = [](energy_t, cycle_t t){return t;};
			break;
		case -1:
			cost_func = [](energy_t e, cycle_t){return e;};
			break;
		default:
			if(ff > 0){
				cost_func = [=](energy_t e, cycle_t t)->cost_t{return pow(e,ff)*t;};
			}else{
				cost_func = [=](energy_t e, cycle_t t)->cost_t{return e*pow(t,-ff);};
			}
	}

/* 
	`stschedule` tool prints input parameters, search time, tree info and cost info.
	These output info are directly writed to the `stdout`. Redirect `stdout` if needed.
 */	
	static const char* mapper_names[] = {"polar", "eyeriss", "systolic", "vector"};  // chess H4: by the first type's core
	cout << "Mapper " << mapper_names[chiplets.type(0).mm];
	cout << " Network " << net_name;
	cout << " Mesh " << (int)Cluster::xlen << '*' << (int)Cluster::ylen;
	cout << " Batch " << tot_batch << endl;
	cout << "compute_die_area: " << compute_die_area << endl;
	cout << "IO_die_area: " << IO_die_area << endl;
	cout << "cost_system_package: " << cost_system_package << endl;
	cout << "cost_overall: " << cost_overall << endl;
	cout << "cost_soc: " << cost_soc << endl;
	chiplets.dump(cout);

#if FORMAT_OUTPUT
	printf("INPUT_PARAMETERS:%s %1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %s %s %3d %s %8d %3d %4d %4d %8d\n", tech.c_str(), mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, package_type.c_str(), IO_type.c_str(), _NoP_bw, DDR_type.c_str(), _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops);
#else
	cout << "INPUT_PARAMETERS: " << tech << ' ' << mm << ' ' << nn << ' ' << xx << ' ' << yy << ' ' 
			<< ss << ' ' << bb << ' ' << rr << ' ' << ff << ' ' 
			<< xcut << ' ' << ycut << ' ' << package_type << ' ' << IO_type << ' ' << _NoP_bw << ' ' 
			<< DDR_type << ' ' << _DRAM_bw << ' ' << _NoC_bw << ' ' << _mac_dim << ' ' 
			<< _ul3 << ' ' << total_tops << endl;
#endif
	Segment_scheme *scheme;

#if FORMAT_OUTPUT
	FILE *file;
	file = argc > 1 ? fopen(argv[1], "a") : fopen("./temp_points.txt", "a");
	auto output = [&](int search_index) {
		fprintf(file, "%s %1d %2d %3d %3d %3d %3d %3d %2d %3d %3d %s %s %3d %s %8d %3d %4d %4d %8d %18.5f %20.5f %10lu %28.5f %32.5f %1d %20.2f %20.2f %20.2f %20.2f %20.2f %20.2f %20.2f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f %10.3f\n", 
				tech.c_str(), mm, nn, xx, yy, ss, bb, rr, ff, xcut, ycut, package_type.c_str(), IO_type.c_str(), _NoP_bw, DDR_type.c_str(), _DRAM_bw, _NoC_bw, _mac_dim, _ul3, total_tops, cost_overall, 
				scheme->schtree_spm->get_cost().energy, scheme->schtree_spm->get_cost().time, scheme->schtree_spm->get_cost().cost(), scheme->schtree_spm->get_cost().cost() * cost_overall, search_index,
				// Energy breakdown
				SchNode::record.ubuf, SchNode::record.buf, SchNode::record.bus, SchNode::record.mac, SchNode::record.NoC_hop_cost, SchNode::record.NoP_hop_cost, SchNode::record.DRAM_cost,
				// Cost breakdown
				compute_die_area, IO_die_area, total_die_area, cost_chip, cost_package, cost_system_package, cost_soc);
	};
#else
	ofstream writer;
	if (argc > 1) {
		writer.open(argv[1], ios::app);
	} else {
		writer.open("./temp.log", ios::app);
	}
	auto output = [&](int search_index) {
		writer << tech << mm << ' ' << nn << ' ' << xx << ' ' << yy << ' ' 
			<< ss << ' ' << bb << ' ' << rr << ' ' << ff << ' ' 
			<< xcut << ' ' << ycut << ' ' << package_type << ' '
			<< IO_type << ' ' << _NoP_bw << ' ' << DDR_type << ' '
			<< _DRAM_bw << ' ' << _NoC_bw << ' ' << _mac_dim << ' ' 
			<< _ul3 << ' ' << total_tops << ' ' << cost_overall << ' ' 
			<< ' ' << scheme->schtree_spm->get_cost().energy << ' ' 
			<< ' ' << scheme->schtree_spm->get_cost().time << ' '
			<< scheme->schtree_spm->get_cost().cost() << ' ' 
			<< scheme->schtree_spm->get_cost().cost() * cost_overall << ''
			<< search_index << endl;
	};
#endif

	// Gemini mapping.
	mode_control(false, false, true);
	scheme = DP_search(SAEngine::nrounds, false, false);
	if (scheme != nullptr) {
		output(3);
		delete scheme;
	}

#if FORMAT_OUTPUT
	fclose(file);
#else
	writer.close();
#endif
	cout << endl << endl << endl;
	return 0;
}
