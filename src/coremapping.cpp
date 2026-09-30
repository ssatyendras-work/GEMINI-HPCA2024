#include "coremapping.h"

#include <cassert>
#include <string>
#include <sstream>
#include <cmath>
#include "partition.h"
#include "util.h"
// #include "Python.h"

#include <fstream>

#define HLEN(num_ol) (wl.H + (wl.R - wl.sH)*(num_ol))
#define WLEN(num_ol) (wl.W + (wl.S - wl.sW)*(num_ol))

#define HTOTL(num_ol) HLEN((num_ol)*pnum_h-1)
#define WTOTL(num_ol) WLEN((num_ol)*pnum_w-1)

#define PolarInst PolarMapper::Instance
#define EyerissInst EyerissMapper::Instance

const char PolarInst::Part::partName[]={
	'C','K','H','W','N',
};

const PolarInst::Part PolarInst::all_parts[]={
	{PartDim::C,PartDim::K},
	{PartDim::C,PartDim::B},
	{PartDim::B,PartDim::K},
	{PartDim::C,PartDim::W},
	{PartDim::H,PartDim::K},
	{PartDim::H,PartDim::W},
};




///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////
xgb_extractor::xgb_extractor(const std::string& model_path) {
	// Create and load the model
	XGBoosterCreate(nullptr, 0, &booster);
	XGBoosterLoadModel(booster, model_path.c_str());
	XGBoosterSetParam(booster, "nthread", "1");
}

std::vector<float> xgb_extractor::extract(DMatrixHandle dmat) {
	bst_ulong out_len;
	const float* out_result;
	XGBoosterPredict(booster, dmat, 2, 0, 0, &out_len, &out_result);
	std::vector<float> features(out_result, out_result + out_len);
    return features;
}

nn_predictor::nn_predictor(const std::string& prefix, int in_dim, int hid_dim, int out_dim, int leaf_num)
    : input_dim(in_dim), hidden_dim(hid_dim), output_dim(out_dim), n_leafs(leaf_num)  {
    load_weights(prefix + "/W1.bin", W1, input_dim * hidden_dim * leaf_num);
    load_weights(prefix + "/b1.bin", b1, hidden_dim);
    load_weights(prefix + "/W2.bin", W2, hidden_dim * output_dim);
    load_weights(prefix + "/b2.bin", b2, output_dim);
}

void nn_predictor::load_weights(const std::string& file, std::vector<float>& vec, int expected_size) {
    vec.resize(expected_size);
    std::ifstream in(file, std::ios::binary);
    if (!in) {
        std::cerr << "Failed to open weight file: " << file << std::endl;
        exit(1);
    }
    in.read(reinterpret_cast<char*>(vec.data()), expected_size * sizeof(float));
    in.close();
}

std::vector<float> nn_predictor::predict(const std::vector<float>& input) {
	// option 1: direct regression
	//  - param: b1, w1, bn_gamma, bn_beta, bn_mean, bn_var
    std::vector<float> hidden(hidden_dim);
	// option 2: one-hot encoding
	//  - extra param: n_leaf
	#pragma omp parallel for
    for (int j = 0; j < hidden_dim; ++j) {
        float sum = b1[j];
		for (int t = 0; t < input_dim; ++t) {
			int leaf = input[t];
			int idx = t * n_leafs + leaf;
			sum += W1[j*input_dim*n_leafs + idx];
		}
		// ReLU
        hidden[j] = std::max(0.0f, sum);
    }

	///////////////////// classifier ///////////////////////
    std::vector<float> out(output_dim);
    #pragma omp parallel for
    for (int j = 0; j < output_dim; ++j) {
        float sum = b2[j];
        for (int i = 0; i < hidden_dim; ++i) {
            sum += hidden[i] * W2[j * hidden_dim + i];
        }
        out[j] = sum;
    }
    return out;
}
	
TL_predictors::TL_predictors(const std::string& model_dir) {
	// Assume meta.log has 3 integers: input_dim hidden_dim output_dim
    std::ifstream meta(model_dir + "/meta.log");
    if (!meta) {
        std::cerr << "Failed to open file: meta.log" << std::endl;
        exit(1);
    }
    int in_dim, hid_dim, out_dim, leaf_num;
    meta >> in_dim >> hid_dim >> out_dim >> leaf_num;

	xgb = std::make_unique<xgb_extractor>(model_dir + "/xgb.model");
    nn = std::make_unique<nn_predictor>(model_dir + "/mlp", in_dim, hid_dim, out_dim, leaf_num);
	load_bias(model_dir + "/output_bias.log", output_bias, out_dim);
}

void TL_predictors::run_predictions(const std::vector<float>& core_feature, std::vector<float>& results) {
	DMatrixHandle dmat;
    XGDMatrixCreateFromMat(core_feature.data(), 1, core_feature.size(), -1, &dmat);
    std::vector<float> leaf_features = xgb->extract(dmat);
    XGDMatrixFree(dmat);
    
	std::vector<float> pred = nn->predict(leaf_features);

	results.resize(pred.size());
    for (size_t i = 0; i < pred.size(); ++i) {
        results[i] = pred[i]*output_bias[i];
    }
}

void TL_predictors::load_bias(const std::string& bias_path, std::vector<float>& bias, int expected_size){
	bias.resize(expected_size);
    std::ifstream in(bias_path);
    if (!in) {
        std::cerr << "Failed to open weight file: " << bias_path << std::endl;
        exit(1);
    }
    float value;
	for(int i=0; i<expected_size; i++){
		in >> value;
		bias[i] = value;
	}
    in.close();
}
///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////





// Codes for CoreMapper
///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////
std::vector<std::unique_ptr<TL_predictors>> CoreMapper::predictors_set;
void CoreMapper::setting_predictor_usage(const std::vector<std::string>& model_paths){
	use_predictor=true;
	for (const auto& path_set : model_paths) {
		predictors_set.push_back(std::make_unique<TL_predictors>(path_set));
	}
}
///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////

CoreMapper::CoreMapper(const Core& c):base_core(c){}

const Core& CoreMapper::core() const{
	return base_core;
}

void CoreMapper::set_utime(Layer& l) const{
	if(REF_IS_INSTANCE(l, ConvLayer)){
		set_conv_utime(static_cast<ConvLayer&>(l));
	}else{
		set_lr_utime(static_cast<LRLayer&>(l));
	}
}

void CoreMapper::set_conv_utime(ConvLayer& l) const{
	utime_t t = l.get_num_op();
	t /= base_core.mac_num;
	l.set_utime(t);
}

void CoreMapper::set_lr_utime(LRLayer& l) const{
	utime_t t = l.get_num_op();
	t /= base_core.LR_mac_num;
	l.set_utime(t);
}


// Codes for PolarMapper
PolarMapper::PolarMapper(const PolarCore& _core)
	:CoreMapper(_core), core(_core){
	for(size_t i=0;i<sizeof(PolarInst::all_parts)/sizeof(Instance::Part);++i){
		assert(PolarInst::all_parts[i].dimO != Instance::PartDim::C);
	}
}

CoreMapper::CoreMapping PolarMapper::genMapping(const ConvWl& wl){
	PolarInst instance(core);
	///////////////////////////////////////////////////////////// Addiing Timeloop /////////////////////////////////////////////////////////////
	return instance.genMapping(wl, use_predictor, use_TL, TL_workpath);
	// return instance.genMapping(wl);
	///////////////////////////////////////////////////////////// Addiing Timeloop /////////////////////////////////////////////////////////////
}

void PolarMapper::set_conv_utime(ConvLayer& l) const{
	utime_t t;
	if(REF_IS_INSTANCE(l, GroupConvLayer)){
		GroupConvLayer::Workload gwl = static_cast<GroupConvLayer&>(l).get_workload();
		gwl.C = DIVCEIL(gwl.GC, core.pes.vecSize)*core.pes.vecSize;
		gwl.K = DIVCEIL(gwl.GK, core.pes.laneNum)*core.pes.laneNum;
		gwl.update_op();
		t = gwl.calc_op(1) * gwl.G;
		if(gwl.GC < core.pes.vecSize && gwl.GK < core.pes.laneNum){
			len_t cDup = core.pes.vecSize / gwl.GC;
			len_t kDup = core.pes.laneNum / gwl.GK;
			t /= MIN(cDup, kDup);
		}
	}else{
		ConvLayer::Workload wl = l.get_workload();
		wl.C = DIVCEIL(wl.C, core.pes.vecSize)*core.pes.vecSize;
		wl.K = DIVCEIL(wl.K, core.pes.laneNum)*core.pes.laneNum;
		wl.update_op();
		t = wl.calc_op(1);
	}
	t /= core.mac_num;
	l.set_utime(t);
}

vol_t PolarMapper::get_ubuf_size() const{
	return core.ul3.Size;
}

// Codes for EyerissMapper
EyerissMapper::EyerissMapper(const EyerissCore& _core)
	: CoreMapper(_core), core(_core) {}

CoreMapper::CoreMapping EyerissMapper::genMapping(const ConvWl& wl){
	EyerissInst instance(core);
	///////////////////////////////////////////////////////////// Addiing Timeloop /////////////////////////////////////////////////////////////
	return instance.genMapping(wl, use_predictor, use_TL, TL_workpath);
	// return instance.genMapping(wl);
	///////////////////////////////////////////////////////////// Addiing Timeloop /////////////////////////////////////////////////////////////
}

vol_t EyerissMapper::get_ubuf_size() const{
	return core.ul2.Size;
}







// Codes for main search:

CoreMapper::CoreMapping CoreMapper::genLayerMap(const Layer& layer, const PartSch& part, const FetchSch& fetch, len_t batch_size, bool wgtB){
	if(REF_IS_INSTANCE(layer, ConvLayer)){
		// Conv Layer...
		const ConvLayer& cl=static_cast<const ConvLayer&>(layer);
		ConvWl wl(cl.get_workload(), DIVCEIL(batch_size, part.B));
		wl.K = DIVCEIL(wl.K, part.K);
		wl.H = DIVCEIL(wl.H, part.H);
		wl.W = DIVCEIL(wl.W, part.W);
		if (fetch) {
			wl.B = DIVCEIL(wl.B, fetch.B);
			wl.K = DIVCEIL(wl.K, fetch.K);
			wl.H = DIVCEIL(wl.H, fetch.H);
			wl.W = DIVCEIL(wl.W, fetch.W);
		}
		if(REF_IS_INSTANCE(layer, GroupConvLayer)){
			const GroupConvLayer& gcl=static_cast<const GroupConvLayer&>(cl);
			const auto& gclWl = gcl.get_workload();
			wl.nGroup = DIVCEIL(wl.K, gclWl.GK);
			wl.C = gclWl.GC;
			if(wl.nGroup > 1)
				wl.K = gclWl.GK;
		}
		if(wgtB){
			wl.nGroup *= wl.B;
			wl.B = 1;
		}
		wl.calc_op();
		CoreMapping m = genMapping(wl);
		if (fetch) {
			m *= fetch.size();
		}
		return m;
	}else if(REF_IS_INSTANCE(layer, LRLayer)){
		assert(!wgtB);
		// LR Layer...
		const LRLayer& lrl=static_cast<const LRLayer&>(layer);
		LRLayer::Workload wl = lrl.get_workload();
		batch_size = DIVCEIL(batch_size, part.B);
		wl.K = DIVCEIL(wl.K, part.K);
		wl.H = DIVCEIL(wl.H, part.H);
		wl.W = DIVCEIL(wl.W, part.W);
		if (fetch) {
			batch_size = DIVCEIL(batch_size, fetch.B);
			wl.K = DIVCEIL(wl.K, fetch.K);
			wl.H = DIVCEIL(wl.H, fetch.H);
			wl.W = DIVCEIL(wl.W, fetch.W);
		}
		wl.update_op();
		access_t tot_op = wl.calc_op(batch_size);

		CoreMapping m;
		m.cost.energy = tot_op * base_core.LR_mac_cost;
		m.cost.time = DIVCEIL(tot_op,base_core.LR_mac_num);
		m.buffer = m.noc = m.ubuf = 0;
		m.mac = m.cost.energy;
		m.util = tot_op;
		m.util /= (m.cost.time * base_core.LR_mac_num);
		m.tot_util = m.util;
		if (fetch) {
			m *= fetch.size();
		}
		return m;
	}else{
		assert(false);
		return CoreMapping();
	}
}

// Codes for PolarInst in main search

PolarInst::Instance(const PolarCore& _core):core(_core){
	// Initialize
	hasWL2 = false;
	hasAL2 = false;
	hasWBSD = false;
	hasKBSD = false;
	K_NB.BSD.cnt = -1;
	W_NB.BSD.cnt = -1;
}

CoreMapper::CoreMapping PolarInst::genMapping(const CoreMapper::ConvWl& wl, bool use_predictor, bool use_TL, const std::string TL_workpath){
	best_map = CoreMapping();

	// For later.
	len_t tot_ifmW = (wl.W-1) * wl.sW + wl.S;
	len_t tot_ifmH = (wl.H-1) * wl.sH + wl.R;
	filSize = wl.fil_size(); //wl.R*wl.S*wl.C*wl.K; // Total filter size.
	ofmSize = wl.ofm_size(); //wl.H*wl.W*wl.K; // Total ofmap size.

	// allcst.energy = energy_inf;
	TGroup = wl.nGroup;
	nDup = 1;
	const vol_t al1USize = core.al1.Size / core.pes.vecSize;
	const vol_t ol1USize = core.ol1.Size / core.pes.laneNum;
	const vol_t wl1USize = core.wl1.Size / core.pes.vecSize;
	vol_t al2USize = core.al2.Size;
	vol_t wl2USize = core.wl2.Size;
	if(wl.nGroup > 1){
		len_t cDup = core.pes.vecSize / wl.C;
		len_t kDup = core.pes.laneNum / wl.K;
		nDup = MIN(cDup, kDup);
		if(nDup > 1){
			nDup = MIN(nDup, wl.nGroup);
			al2USize /= nDup;
			wl2USize /= nDup;
			TGroup = DIVCEIL(wl.nGroup, nDup);
		}
	}

	for(size_t i=0;i<sizeof(all_parts)/sizeof(Part);++i){
		const Part& p = all_parts[i];

		// Calculate part. dup. and total loops.
		pnum_c = pnum_k = pnum_h = pnum_w = pnum_b = 1;
		switch (p.dimA) {
			case PartDim::C: pnum_c*=core.bus.aLen; break;
			case PartDim::K: pnum_k*=core.bus.aLen; break;
			case PartDim::H: pnum_h*=core.bus.aLen; break;
			case PartDim::W: pnum_w*=core.bus.aLen; break;
			case PartDim::B: pnum_b*=core.bus.aLen; break;
			default: assert(false);
		}
		switch (p.dimO) {
			case PartDim::C: pnum_c*=core.bus.oLen; break;
			case PartDim::K: pnum_k*=core.bus.oLen; break;
			case PartDim::H: pnum_h*=core.bus.oLen; break;
			case PartDim::W: pnum_w*=core.bus.oLen; break;
			case PartDim::B: pnum_b*=core.bus.oLen; break;
			default: assert(false);
		}
		wdup = pnum_h * pnum_w * pnum_b;
		adup = pnum_k;

		totTC = DIVCEIL(wl.C,core.pes.vecSize * pnum_c);
		totTK = DIVCEIL(wl.K,core.pes.laneNum * pnum_k);
		totOC = DIVCEIL(wl.C,pnum_c);
		totOK = DIVCEIL(wl.K,pnum_k);
		totTH = DIVCEIL(wl.H,pnum_h);
		totTW = DIVCEIL(wl.W,pnum_w);
		totTB = DIVCEIL(wl.B,pnum_b);
		comp_time = totTC * totTK * totTH * totTW * totTB * wl.R * wl.S;

		tot_op_k = (wl.tot_op/wl.K)*totTK*pnum_k;
		tot_op_c = (wl.tot_op/wl.C)*totTC*pnum_c;

		// Calculate L1 size:
		/* AL1: H'1*W'1*(C?)
		 * WL1: R*S*(C?)*K1 (8 WL1 in total)
		 * OL1: H1*W1*(1*8?)
		 * AL2: H'1*(W1*WBSD*W2)'*C*NDUP (H'W' adds duplicate)
		 * [WL2: R*S*C*(K1*KBSD*K2*8)*K_distinct]
		 */
		// Update: now we consider size of each lane/vec unit.
		// AL1: H'1*W'1*(C/8)
		vol_t al1HW = al1USize / totTC;
		// OL1: H1*W1*1
		vol_t ol1HW = ol1USize;
		// WL1: R*S*(C/8)*K1
		len_t K1_max = wl1USize / (wl.R * wl.S * totTC);
		len_t wl2Kmax = wl2USize /
				(wl.R * wl.S * wl.C * pnum_k * core.pes.laneNum);
		len_t al2HWDup = al2USize / (wl.C * MIN(pnum_b,wl.B));
		// Check whether L1 cstrs (hard) are satisfied:
		len_t max_w1 = al1HW/wl.R;
		if(ol1HW == 0 || K1_max == 0 || max_w1 < wl.S){
			// Invalid partition since no sufficient L1.
			continue;
		}
		max_w1 = (max_w1 - wl.S)/wl.sW + 1;
		max_w1 = MIN(max_w1, ol1HW);
		max_w1 = MIN(max_w1, totTW);

		// H1*W1 <= ol1HW, H'1*W'1 <= al1HW, K1 <= K1_max
		//
		// K1*KBSD*K2 <= wl2Kmax
		// Or
		// H'1d*(W1*WBSD*W2)'d (counting dup.) <= al2HWDup
		//
		// If no WL2/AL2, then wl2Kmax/al2HWDup = 0.

		/* Notice: totOC * 16 >= C * pnum_k * wdup
		 * Then if wl2Size >= 16*8*wl1Size
		 * Then: wl2Kmax / wdup
		 *  = wl2Size / (R * S * C * pnum_k * 8 * wdup)
		 * >= 16*wl1Size / (R * S * C * pnum_k * wdup)
		 * >= wl1Size / (R * S * totOC)
		 *  = K1_max
		 *
		 * Thus wl2Kmax >= K1_max * wdup >= K1 * KBSD
		 * Guarantees K2 >= 1
		 *
		 * Similarly:
		 * al2HWDup = al2Size / (wl.C * MIN(pnum_n,N))
		 * al1HW = al1Size / totOC
		 * H'1d <= pnum_h * H'1
		 * H'1d*(W1*WBSD*W2)'d <= al2HWDup
		 * H'1*W'1 <= al1HW
		 * totOC * 16 >= C * pnum_k * pnum_h * pnum_w * pnum_n
		 * Guarantees:
		 * al2HWDup / H'1d >= (W'1 * pnum_w) * pnum_k
		 * Guarantees (W1*WBSD*W2)'d >= W'1d * pnum_k
		 */
		K1.tot_above = totTK;
		H1.tot_above = totTH;
		W1.tot_above = totTW;

		// Init K1, K_B, K_NB
		K1.cnt = MIN(totTK, K1_max);
		K_NB.BSD.tot_above = K_B.BSD.tot_above =
		K_NB.L2.tot_above = K_NB.L3_N2.tot_above =
		K_NB.L3_N2.cnt =
		DIVCEIL(totTK, K1.cnt);

		// K_NB: No BSD
		// For DataFlow::KHWN
		if(wl2Kmax > 0){
			if(totTK > wl2Kmax){
				K_NB.L2.cnt = wl2Kmax / K1.cnt;
				K_NB.L3.cnt = K_NB.L3.tot_above = DIVCEIL(K_NB.L2.tot_above, K_NB.L2.cnt);
			}else{
				K_NB.L2.cnt = K_NB.L2.tot_above;
				K_NB.L3.cnt = K_NB.L3.tot_above = 1;
			}
		}else{
			K_NB.L2.cnt = 1;
			K_NB.L3.cnt = K_NB.L3.tot_above = K_NB.L2.tot_above;
		}

		// With BSD.
		K_B.BSD.cnt = MIN(K_B.BSD.tot_above, wdup);
		if(K_B.BSD.cnt > 1){
			K_B.L3_N2.cnt = K_B.L3_N2.tot_above = K_B.L2.tot_above = DIVCEIL(K_B.BSD.tot_above, K_B.BSD.cnt);

			// For DataFlow::KHWN
			if(wl2Kmax > 0){
				if(totTK > wl2Kmax){
					K_B.L2.cnt = wl2Kmax / (K1.cnt * K_B.BSD.cnt);
					K_B.L3.cnt = K_B.L3.tot_above = DIVCEIL(K_B.L2.tot_above, K_B.L2.cnt);
				}else{
					K_B.L2.cnt = K_B.L2.tot_above;
					K_B.L3.cnt = K_B.L3.tot_above = 1;
				}
			}else{
				K_B.L2.cnt = 1;
				K_B.L3.cnt = K_B.L3.tot_above = K_B.L2.tot_above;
			}
		}

		bool tn_not_one = (totTB > 1);
		cycle_t tile_time_buf = wl.R * wl.S * totTC * K1.cnt;
		wl1TileSize = wl.R * wl.S * totOC * MIN(totOK, K1.cnt * core.pes.laneNum);
		// Search for appropriate H,W
		len_t last_h1 = 0;
		len_t aw1 = max_w1*wl.sW + wl.S;
		for(len_t w1=max_w1;w1>=1;--w1){
			aw1 -= wl.sW;
			//aw1 = (w1-1)*wl.sW + wl.S;
			len_t h1 = ol1HW / w1;
			len_t ah1_hmax = al1HW/aw1;
			// We must have al1HW/aw1 >= wl.R
			ah1_hmax = (ah1_hmax- wl.R)/wl.sH+1;
			h1 = MIN(h1, ah1_hmax);
			h1 = MIN(h1, totTH);
			if(last_h1 == h1) continue;
			last_h1 = h1;

			H1.cnt = h1;
			H3.cnt = H3.tot_above = DIVCEIL(totTH, h1);
			W1.cnt = w1;

			tile_time = tile_time_buf * h1 * w1;

			bool nh_not_one = tn_not_one || (H3.cnt > 1);

			W_NB.BSD.tot_above = W_B.BSD.tot_above =
			W_NB.L2.tot_above = W_NB.L3_N2.tot_above =
			W_NB.L3_N2.cnt = DIVCEIL(totTW, w1);

			// H'1d*(W1*WBSD*W2)'d (counting dup.) <= al2HWDup
			len_t adh1 = (h1 == totTH)?tot_ifmH:((h1-1)*wl.sH + wl.R)*pnum_h;
			// As shown before, tot_w >= W'1d * pnum_k
			len_t tot_w = al2HWDup / adh1;
			if(al2HWDup > 0){
				tot_w = (tot_w >= tot_ifmW)?totTW:(tot_w/pnum_w-wl.S)/wl.sW+1;

				if(totTW > tot_w){
					W_NB.L2.cnt = tot_w / W1.cnt;
					W_NB.L3.cnt = W_NB.L3.tot_above = DIVCEIL(W_NB.L2.tot_above, W_NB.L2.cnt);
				}else{
					W_NB.L2.cnt = W_NB.L2.tot_above;
					W_NB.L3.cnt = W_NB.L3.tot_above = 1;
				}
			}else{
				W_NB.L2.cnt = 1;
				W_NB.L3.cnt = W_NB.L3.tot_above = W_NB.L2.tot_above;
			}

			W_B.BSD.cnt = MIN(W_B.BSD.tot_above, adup);
			if(W_B.BSD.cnt > 1){
				W_B.L3_N2.cnt = W_B.L3_N2.tot_above = W_B.L2.tot_above = DIVCEIL(W_B.BSD.tot_above, W_B.BSD.cnt);

				// For DataFlow::HWNK
				if(al2HWDup > 0){
					if(totTW > tot_w){
						W_B.L2.cnt = tot_w / W1.cnt * W_B.BSD.cnt;
						W_B.L3.cnt = W_B.L3.tot_above = DIVCEIL(W_B.L2.tot_above, W_B.L2.cnt);
					}else{
						W_B.L2.cnt = W_B.L2.tot_above;
						W_B.L3.cnt = W_B.L3.tot_above = 1;
					}
				}else{
					W_NB.L2.cnt = 1;
					W_B.L3.cnt = W_B.L3.tot_above = W_B.L2.tot_above;
				}
			}

			// DataFlow::KHWN
			curDF = DataFlow::KHWB;
			// hasAL2 = false;
			// hasWL2 = false;
			// hasWBSD = false;
			// hasKBSD = false;
			// A: No BSD, L2
			//allcst.energy = hasl2.energy = -1;
			getCost(wl);

			if(nh_not_one && W_NB.L3_N2.tot_above > 1){
				if(K_NB.L2.cnt > 1){
					// B: No BSD, has L2
					hasWL2 = true;
					getCost(wl);
					hasWL2 = false;
				}
				if(K_B.BSD.cnt > 1){
					// C: No WBSD, WL2, has KBSD
					hasKBSD = true;
					getCost(wl);
				}
			}
			// Here hasWL2 = hasWBSD = false;
			if(W_B.BSD.cnt > 1){
				hasWBSD = true;
				if(K_B.BSD.cnt > 1 && K_B.L3_N2.tot_above > 1){
					// E: No WL2, has BSD
					hasKBSD = true;
					getCost(wl);
				}
				if(nh_not_one && W_B.L3_N2.tot_above > 1){
					hasWL2 = true;
					if(K_NB.L3.tot_above > 1){
						// D: No KBSD, has WBSD, WL2
						hasKBSD = false;
						getCost(wl);
					}
					if(K_B.BSD.cnt > 1 && K_B.L3.tot_above > 1){
						// F: Has BSD, L2
						hasKBSD = true;
						getCost(wl);
					}
					hasWL2 = false;
				}
				hasWBSD = false;
			}
			hasKBSD = false;

			// DataFlow::HWNK (Similar to KHWN)
			curDF = DataFlow::HWBK;

			//_try_print(wl, i);

			//printf("%s %d*%d %.2e\n",p.getName().c_str(),H1.cnt,W1.cnt,cost.energy);
		}
	}
//	std::cout << "time= " << allcst.time << "energy=" << allcst.energy << "\n";
	return best_map;
}

void PolarInst::getCost(const ConvWl& wl){
	// (X)L2D Loop: The first relevant (to X) loop above L1/BSD
	// (X)L2T Loop: The first relevant (to X) loop above L2
	// (X)L1T Loop: The first relevant (to X) loop above L1

	len_t KBT, K2T, K3T, WBT, W3T;
	const DLoops& KL = hasKBSD?K_B:K_NB;
	const DLoops& WL = hasWBSD?W_B:W_NB;
	if(hasWL2){
		K3T = KL.L3.tot_above;
		K2T = KL.L2.tot_above;
	}else{
		K3T = K2T = KL.L3_N2.tot_above;
	}
	KBT = KL.BSD.tot_above;
	W3T = hasAL2?WL.L3.tot_above:WL.L3_N2.tot_above;
	WBT = WL.BSD.tot_above;

	access_t WBA = WBT*H3.tot_above*totTB;
	access_t W3A = W3T*H3.tot_above*totTB;

	// Definition of variables for calculation.
	vol_t a1wIfmSize; // Ifmap size, including overlaps above AL1T. (include Part.)
	access_t a1wFetch; // Fetch times above AL1T;

	access_t w1wFetch; // Fetch times above WL1T;
	access_t w1rFetch; // Fetch times above WL1(D);

	vol_t a2wIfmSize; // Ifmap size, including overlaps above AL2T. (include Part.)
	vol_t a2rIfmSize; // Ifmap size, including overlaps above AL2D. (include Part.)
	access_t a2wFetch; // Fetch times above AL2T; (Only ava. if has AL2)
	access_t a2rFetch; // Fetch times above AL2D;

	access_t w2wFetch; // Fetch times above WL2T; (Only ava. if has WL2)
	access_t w2rFetch; // Fetch times above WL2D;

	//access_t al1ReadBSD;
	access_t al1WriteBSD=0; // Overlap introduces excessive write.
	access_t al1BSDShare=0;
	access_t wl1BSDShare=0;

	access_t aBSDTot = 0;
	access_t wBSDTot = 0;

	switch (curDF) {
	case DataFlow::KHWB:{
		vol_t tmpSize = HTOTL((WBT == 1)?1:H3.tot_above) * wl.C * wl.B;
		a1wIfmSize = tmpSize * WTOTL(hasWBSD?W3T:1); // Ifmap size, including overlaps above AL1T. (include Part.)
		a1wFetch = hasWBSD?K2T:K3T; // Fetch times above AL1T;

		w1rFetch = WBA;
		w1wFetch = hasKBSD?WBA:(hasWL2?W3A:1); // Fetch times above WL1T;

		a2wIfmSize = 0;//HTOTL(H3.tot_above) * WTOTL(1) * wl.C * wl.B; // Ifmap size, including overlaps above AL2T. (include Part.)
		a2rIfmSize = HTOTL((W3T == 1)?1:H3.tot_above) * WTOTL(1) * wl.C * wl.B; // Ifmap size, including overlaps above AL2D. (include Part.)

		a2wFetch = 0;// Fetch times above AL2T;
		a2rFetch = (hasKBSD&&hasWBSD&&(!hasWL2))?1:K3T; // Fetch times above AL2D;

		w2rFetch = (hasWBSD || hasWL2)? W3A:1; // Fetch times above WL2D;
		w2wFetch = 1; // Fetch times above WL2T;

		if(hasWBSD){
			vmac_t tmplen = ((adup-1)*WBT) % adup;
			al1WriteBSD = tmpSize * (wl.S - wl.sW) * pnum_w * a1wFetch;
			al1BSDShare = (tmpSize * (wl.W - (W1.cnt * (WBT - (WBT % adup))+ wl.sW - wl.S)*pnum_w) *a1wFetch + al1WriteBSD) * tmplen;
			al1WriteBSD *= W3T*(adup - 1); // Overlap introduces excessive write.
		}
		if(hasKBSD)
			wl1BSDShare = (filSize / wl.K) * (wl.K - K1.cnt*core.pes.laneNum*pnum_k*(KBT-KBT%wdup)) *w1wFetch * (((wdup-1)*KBT) % wdup);
	}break;
	case DataFlow::HWBK:
		a1wIfmSize=a1wFetch=
		w1wFetch=w1rFetch=
		a2wIfmSize=a2rIfmSize=
		a2wFetch=a2rFetch=
		w2wFetch=w2rFetch=0;
		assert(false);
	break;
	default:
		a1wIfmSize=a1wFetch=
		w1wFetch=w1rFetch=
		a2wIfmSize=a2rIfmSize=
		a2wFetch=a2rFetch=
		w2wFetch=w2rFetch=0;
		assert(false);
	}

	access_t al2Read =  a2rIfmSize * a2rFetch;
	access_t al2Write = hasAL2 ? a2wIfmSize*a2wFetch : al2Read;

	access_t wl2Read = filSize * w2rFetch;
	access_t wl2Write = hasWL2 ? filSize*w2wFetch : wl2Read;

	access_t al1Read = tot_op_k; // Add tot_op using totTK as K
	access_t al1Write = a1wIfmSize * a1wFetch * adup  + al1WriteBSD;
	if(hasWBSD){
		aBSDTot = al1Write - al2Read;
		al1Read += aBSDTot - al1BSDShare;
	}

	access_t wl1Read = filSize * w1rFetch * wdup;
	access_t wl1Write = filSize * w1wFetch * wdup;
	if(hasKBSD){
		wBSDTot = wl1Write - wl2Read;
		wl1Read += wBSDTot - wl1BSDShare;
	}

	access_t ol1Write = tot_op_c;
	access_t ol1Read = tot_op_c + pnum_c*ofmSize;

	access_t ol2Read = ofmSize;
	access_t ol2Write = ofmSize;

	access_t ul3Read = al2Write + wl2Write;
	access_t ul3Write = ofmSize;

	energy_t buf_energy=0, noc_energy=0, mac_energy=0;
	buf_energy = al1Read * core.al1.RCost
			+ al1Write * core.al1.WCost
			+ wl1Read * core.wl1.RCost
			+ wl1Write * core.wl1.WCost
			+ ol1Read * core.ol1.RCost
			+ ol1Write * core.ol1.WCost
			+ ol2Read * core.ol2.RCost
			+ ol2Write * core.ol2.WCost;
	energy_t ubuf_energy =
			+ ul3Read * core.ul3.RCost
			+ ul3Write * core.ul3.WCost;
	if(hasAL2){
		buf_energy += al2Read * core.al2.RCost;
		buf_energy += al2Write * core.al2.WCost;
	}
	if(hasWL2){
		buf_energy += wl2Read * core.wl2.RCost;
		buf_energy += wl2Write * core.wl2.WCost;
	}
	if(hasKBSD) noc_energy += wBSDTot * core.bus.hopCost;
	if(hasWBSD) noc_energy += aBSDTot * core.bus.hopCost;
	mac_energy = wl.tot_op * core.pes.MACCost;
	energy_t tot_energy = ubuf_energy + buf_energy + noc_energy + mac_energy;
/*
	printf("%.3f\t",((double)al1Read * core.al1.RCost)/tot_energy);
	printf("%.3f\t",((double)al1Write * core.al1.WCost)/tot_energy);
	printf("%.3f\t",((double)wl1Read * core.wl1.RCost)/tot_energy);
	printf("%.3f\t",((double)wl1Write * core.wl1.WCost)/tot_energy);
	printf("%.3f\t",((double)ol1Read * core.ol1.RCost)/tot_energy);
	printf("%.3f\t",((double)ol1Write * core.ol1.WCost)/tot_energy);
	printf("%.3f\t",((double)ol2Read * core.ol2.RCost)/tot_energy);
	printf("%.3f\t",((double)ol2Write * core.ol2.WCost)/tot_energy);
	printf("%.3f\t",((double)ul3Read * core.ul3.RCost)/tot_energy);
	printf("%.3f\t",((double)ul3Write * core.ul3.WCost)/tot_energy);
	if(hasWL2){
		printf("%.3f\t",((double)wl2Read * core.wl2.RCost)/tot_energy);
		printf("%.3f\n",((double)wl2Write * core.wl2.WCost)/tot_energy);
	}else{
		printf("0\t0\n");
	}
*/
	if(hasKBSD){
		cycle_t cyc = (curDF == DataFlow::HWBK && hasWBSD)?WL.BSD.cnt*tile_time:tile_time;
		if(cyc*core.bus.busBW < wl1TileSize * nDup){
			return;
		}
	}
	if(hasWBSD){
		cycle_t cyc = (curDF == DataFlow::KHWB && hasKBSD)?KL.BSD.cnt*tile_time:tile_time;
		if(cyc*core.bus.busBW < (H1.cnt + wl.R - wl.sH) * W1.cnt * totOC * nDup){
			return;
		}
	}

	cycle_t tot_time = comp_time;
	cycle_t ul3_rtime = DIVCEIL(ul3Read * nDup,core.ul3.RBW);
	tot_time = MAX(tot_time, ul3_rtime);
	if(hasAL2){
		cycle_t al2_rtime = DIVCEIL(al2Read * nDup,core.al2.RBW);
		cycle_t al2_wtime = DIVCEIL(al2Write * nDup,core.al2.WBW);
		tot_time = MAX(tot_time, al2_rtime);
		tot_time = MAX(tot_time, al2_wtime);
	}
	if(hasWL2){
		cycle_t wl2_rtime = DIVCEIL(wl2Read * nDup,core.wl2.RBW);
		cycle_t wl2_wtime = DIVCEIL(wl2Write * nDup,core.wl2.WBW);
		tot_time = MAX(tot_time, wl2_rtime);
		tot_time = MAX(tot_time, wl2_wtime);
	}
	// Ignore tail time where needs smaller BW.
	// i.e. consider 30 group and 4 nDup.
	if(wl.nGroup > 1){
		tot_energy *= wl.nGroup;
		ubuf_energy *= wl.nGroup;
		buf_energy *= wl.nGroup;
		noc_energy *= wl.nGroup;
		mac_energy *= wl.nGroup;
		tot_time *= TGroup;
	}
	double tot_op = wl.tot_op * wl.nGroup;
	double util = tot_op / (comp_time * TGroup * core.mac_num);
	assert(util <= 1 + 1e-6);
	double tot_util = tot_op / (tot_time * core.mac_num);
	/*printf("%llu\n",al2_rtime);
	printf("%llu\n",al2_wtime);
	printf("%llu\n",ol2_rtime);
	printf("%llu\n",ol2_wtime);
	printf("%llu\n",wl2_rtime);
	printf("%llu\n",wl2_wtime);
	printf("%llu\n",ul3_rtime);
	printf("%llu\n",ul3_wtime);*/
	MapCost cost(tot_energy, tot_time);
	if(cost.cost() < best_map.cost.cost()){
		best_map = {cost, ubuf_energy, buf_energy, noc_energy, mac_energy, util, tot_util};
	}
	return;
}


///////////////////////////////////////////////////////////// Addiing Timeloop function /////////////////////////////////////////////////////////////
void PolarInst::write_TL_hyperparams(const ConvWl& wl, const std::string working_path){
	// todo: TL to support NVDLA
}

CoreMapper::CoreMapping PolarInst::read_TL_cost(const ConvWl& wl, const std::string working_path){
	// todo: TL to support NVDLA
}
///////////////////////////////////////////////////////////// Addiing Timeloop function /////////////////////////////////////////////////////////////





// Codes for EyerissInst in main search

EyerissInst::Instance(const EyerissCore& _core):core(_core){}

/*
 * reply_h = 32
 * maxc = 3
 * maxk = 4
 * 3 * 4 * 2
 * 2 * 4 * 4
 * 1 * 4 * 8
 *
 * (3 * 4 * 2)
 * 3 * 3 * 3
 * 3 * 2 * 5
 * 3 * 1 * 10
 *
 * 32
 * 3
 * 11
 * 2 * 11 * 1
 * 3 * 10 * 1
*/




///////////////////////////////////////////////////////////// Addiing Timeloop function /////////////////////////////////////////////////////////////
CoreMapper::CoreMapping EyerissInst::genMapping(const CoreMapper::ConvWl& wl, bool use_predictor, bool use_TL, const std::string TL_workpath) {
	best_map = CoreMapping();
	
	if(use_TL){
		write_TL_hyperparams(wl, TL_workpath);
		std::string command = "python3 ./TL_predictor/gen_TL_data/run_TL_in_Gemini.py";
		command = command + " --working_path " + TL_workpath;
		command += " > " + TL_workpath + "/temp.log 2>&1";
    	std::system(command.c_str());
		best_map = read_TL_cost(wl, TL_workpath);
	}
///////////////////////////////////////////////////////////// Addiing Timeloop function /////////////////////////////////////////////////////////////

	else{
		///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////
		bool has_predictor = false;
		int predictor_id;
		if ((wl.sW==1) && (wl.sH==1)){
			if ((wl.S==3) && (wl.R==3) ) {  // case 1: 3x3 conv
				predictor_id = 0;
				has_predictor = true;
			}
			else if ((wl.S==1) && (wl.R==1) && (wl.W==1)){  // case 2: gemm
				predictor_id = 1;
				has_predictor = true;
			}
			else if ((wl.S==1) && (wl.R==1)){  // case 3: 1x1 conv (not matmul)
				if (wl.K <= 512){
					predictor_id = 2;
					has_predictor = true;
				}
			}
		}
		if(has_predictor && use_predictor){
			// if (predictor_id == 0){
			// 	std::cout << "using conv_k3 predictor \n";
			// }
			// else if (predictor_id == 1){
			// 	std::cout << "using gemm predictor \n";
			// }
			// else if (predictor_id == 2){
			// 	std::cout << "using conv_k1 predictor \n";
			// }
			std::vector<float> pred_inputs = gen_predictor_input(wl, core);
			std::vector<float> pred_results;
			predictors_set[predictor_id]->run_predictions(pred_inputs, pred_results);
			// # tot_util, E_weights, E_ifmap, E_psum, E_glb
			energy_t ubuf_energy=0, buf_energy=0, noc_energy=0, mac_energy=0;
			int layout_bias = 0*5;
			ubuf_energy = pred_results[layout_bias+4] * wl.tot_op / 1000;
			buf_energy  = (pred_results[layout_bias+1] + pred_results[layout_bias+2] + pred_results[layout_bias+3]) * wl.tot_op / 1000;
			noc_energy  = 0;
			mac_energy  = 0.17308 * wl.tot_op;

			energy_t tot_energy = ubuf_energy + buf_energy + noc_energy + mac_energy;

			double util = pred_results[layout_bias+0];
			util = MAX(0.01, util);
			util = MIN(1, util);
			double exp_time = wl.tot_op / (util * core.pes.Xarray*core.pes.Yarray);
			cycle_t tot_time = exp_time;
			double tot_util = util;
			
			MapCost cost(tot_energy, tot_time);
			best_map = {cost, ubuf_energy, buf_energy, noc_energy, mac_energy, util, tot_util};
		}
		
	///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////
		// else using the old heuristics
		else{
			// std::cout << "using Gemini heuristics \n";
			tot_ifmW = (wl.W - 1) * wl.sW + wl.S;
			tot_ifmH = (wl.H - 1) * wl.sH + wl.R;
			ifmSize = wl.ifm_size(); //tot_ifmW * tot_ifmH * wl.C * wl.B;
			filSize = wl.fil_size(); //wl.R * wl.S * wl.C * wl.K; // Total filter size.
			ofmSize = wl.ofm_size(); //wl.H * wl.W * wl.K * wl.B; // Total ofmap size.
			_al1_buf = wl.R * wl.C * wl.H * wl.B * tot_ifmW;

			assert(core.pes.Yarray >= wl.R);
			// Fold & replicate
			fold_t fold_w = DIVCEIL(wl.H, core.pes.Xarray);
			phyarr_w = static_cast<vmac_t>(DIVCEIL(wl.H, fold_w));

			reply_t k_reply_w = MAX(core.pes.Xarray / wl.H, 1);
			k_reply_w = MIN(wl.K, k_reply_w);
			reply_t reply_h = core.pes.Yarray / wl.R;

			len_t remain_k = DIVCEIL(wl.K, k_reply_w);

			len_t total_b;
			max_c1 = core.al1.Size / wl.S;
			max_k1 = core.wl1.Size / (wl.S * max_c1);
			total_b = wl.B * fold_w;

			reply_t max_kreplh, max_creplh;
			max_kreplh = DIVCEIL(remain_k, max_k1);
			max_kreplh = MIN(max_kreplh, reply_h);
			max_creplh = DIVCEIL(wl.C, max_c1);
			max_creplh = MIN(max_creplh, reply_h);

			reply_t c_reply_h, k_reply_h;

			if(max_creplh * max_kreplh >= reply_h){
				b_reply = 1;
				Bt = total_b;
				c_reply_h = reply_h / max_kreplh;
				k_reply_h = reply_h / c_reply_h;
				// Can prove now c_reply_h = reply_h / k_reply_h;
				do{
					// Using c&k
					Ct = DIVCEIL(wl.C, c_reply_h);
					Kt = DIVCEIL(remain_k, k_reply_h);
					k_reply = k_reply_w * k_reply_h;
					getCost(wl);
					if(c_reply_h >= max_creplh) break;
					k_reply_h = reply_h / (c_reply_h + 1);
					c_reply_h = reply_h / k_reply_h;
				}while(true);
			}else if(max_creplh * max_kreplh * total_b >= reply_h){
				reply_t max_breplh, rem_replh; // remain repl. h
				// Fix k.
				// k_reply_h = max_kreplh;
				Kt = DIVCEIL(remain_k, max_kreplh);
				k_reply = k_reply_w * max_kreplh;
				rem_replh = reply_h / max_kreplh;

				max_breplh = MIN(total_b, rem_replh);
				c_reply_h = rem_replh / max_breplh;
				b_reply = rem_replh / c_reply_h;
				// Can prove now c_reply_h = rem_replh / n_reply;
				do{
					// Using c&n
					Ct = DIVCEIL(wl.C, c_reply_h);
					Bt = DIVCEIL(total_b, b_reply);
					getCost(wl);
					if(c_reply_h >= max_creplh) break;
					b_reply = rem_replh / (c_reply_h + 1);
					c_reply_h = rem_replh / b_reply;
				}while(true);

				// Fix c.
				// c_reply_h = max_creplh;
				Ct = DIVCEIL(wl.C, max_creplh);
				rem_replh = reply_h / max_creplh;

				max_breplh = MIN(total_b, rem_replh);
				k_reply_h = rem_replh / max_breplh;
				b_reply = rem_replh / k_reply_h;
				// Can prove now k_reply_h = rem_replh / n_reply;
				do{
					// Using k&n
					Kt = DIVCEIL(remain_k, k_reply_h);
					Bt = DIVCEIL(total_b, b_reply);
					k_reply = k_reply_w * k_reply_h;
					getCost(wl);
					//std::cout<<k_reply_h<<' '<<n_reply<<std::endl;
					if(k_reply_h >= max_kreplh) break;
					b_reply = rem_replh / (k_reply_h + 1);
					k_reply_h = rem_replh / b_reply;
				}while(true);
			}else{
				Ct = DIVCEIL(wl.C, max_creplh);
				Kt = DIVCEIL(remain_k, max_kreplh);
				k_reply = k_reply_w * max_kreplh;
				b_reply = total_b;
				Bt = 1;
				assert(Ct <= max_c1);
				assert(Kt <= max_k1);
				getCost(wl);
			}
		}
	}
	best_map *= wl.nGroup;
	return best_map;
}

/*
 * For Loop:
 *
 * for K2
 *  for C2
 *   for Nt
 *    for W
 *     for K1
 *      for C1
 *       for S
 *
 * L1 size req.:
 * IFM: C1*S
 * FIL: C1*K1*S
 * OFM: 1 (of psum width)
 *
 * Fold: W
 * Replicate: CKN(H),K(W)
 * Physical Arr: R*PW
 *     PW = DIVCEIL(H, foldW)
 *
 * UL2 access:
 * FIL-W/IFM-W: #FIL/#IFM (May be added outside)
 * FIL-R: #FIL
 * IFM-R: K2*(Hifm + (R - sH)*(foldW - 1))*Wifm*C*N
 * OFM-R/W: C2 * #OFM
 *
 * ITCN access:
 * X = FIL/IFM
 * X-W = X-R @ UL2
 * X-R = X-W @ XL1
 * OFM-R/W = (OFM-W + OFM-R) @ UL2 - #OFM
 * // P.S. Notice that psums never bcast
 *
 * L1 access:
 * FIL-R: op
 * FIL-W: #FIL * replN * PW
 *     // Actual size may be slightly smaller when Nt is very small and PH*foldW>H.
 * IFM-R: op
 * IFM-W: Wifm*R*C*H*N*DIVCEIL(K,K1)
 *     // The "DIVCEIL(K,K1)" term may be a little larger. (See getCost)
 * OFM-R: op - #OFM
 * OFM-W: op - #OFM
 *
 * Time: Nt*Ct*Kt*W*S
 *
 * Variables needed for energy:
 * replN, K1, K2, C2
 */\

void EyerissInst::getCost(const CoreMapper::ConvWl& wl) {
	access_t al1_r, al1_w, wl1_r, wl1_w, pl1_r, pl1_w, ul2_r, ul2_w; // ul2_w will be calculated out of intra-core cost.
	access_t ul2_ar, ul2_aw, ul2_wr, ul2_ww, ul2_pr, ul2_pw;
	access_t bus_aw, bus_ww, bus_pw;
	cycle_t d_ibus, d_wbus, d_pbus;
	cycle_t comp_time;

	len_t C1 = MIN(Ct, max_c1);
	len_t C2 = DIVCEIL(Ct, C1);
	len_t K1 = MIN(Kt, max_k1);
	len_t K2 = DIVCEIL(Kt, K1);

	comp_time = wl.W * wl.S * Kt * Ct * Bt;

	//for l2
	ul2_ww = 0;
	ul2_wr = /*ul2_ww =*/ filSize;
	ul2_pr = ul2_pw = ofmSize * C2;
	ul2_ar = K2 * wl.C * wl.B;
	ul2_aw = 0; //ifmSize;
	ul2_r = ul2_wr + ul2_ar + ul2_pr;
	ul2_w = ul2_ww + ul2_aw + ul2_pw;

	// for interconnect
	bus_ww = ul2_wr;
	bus_aw = ul2_ar;
	bus_pw = ul2_pr + ul2_pw - ofmSize;
	d_wbus = DIVCEIL(bus_ww, core.wbus.BusBW);
	d_ibus = DIVCEIL(bus_aw, core.ibus.BusBW);
	d_pbus = DIVCEIL(bus_pw, core.pbus.BusBW); // Buffer bandwidth is equal to bus bandwidth

	// for l1
	wl1_r = al1_r = wl.tot_op;
	pl1_r = pl1_w = wl.tot_op - ofmSize;

	wl1_w = filSize * b_reply * phyarr_w;

	// 300: 3/10/10 -> 30
	// 306: 4/10/10 -> 36
	// 325: 4/10/10 -> 40
	len_t al1_k = K2 * k_reply;
	if(Kt % K1 == 1) al1_k -= k_reply - (wl.K % k_reply);
	al1_w = _al1_buf * al1_k;

	energy_t buf_energy=0, noc_energy=0, mac_energy=0, ubuf_energy=0;

	buf_energy =  al1_w * core.al1.WCost
				+ wl1_w * core.wl1.WCost
				+ pl1_w * core.pl1.WCost
				+ al1_r * core.al1.RCost
				+ wl1_r * core.wl1.RCost
				+ pl1_r * core.pl1.RCost;
	ubuf_energy = ul2_w * core.ul2.WCost
				+ ul2_r * core.ul2.RCost;

	noc_energy =  bus_aw * core.ibus.BusCost
				+ bus_ww * core.wbus.BusCost
				+ bus_pw * core.pbus.BusCost;

	mac_energy =  wl.tot_op * core.pes.MacCost;

	energy_t tot_energy = ubuf_energy + buf_energy + noc_energy + mac_energy;

	cycle_t tot_time;
	tot_time = MAX(d_ibus, d_pbus);
	tot_time = MAX(tot_time, d_wbus);
	tot_time = MAX(tot_time, comp_time);

	double util = wl.tot_op * 1.0 / (comp_time * core.mac_num);
	//std::cout<<util<<' '<<Kt<<' '<<Ct<<' '<<Nt<<std::endl;
	//std::cout<<' '<<wl.K<<' '<<wl.C<<' '<<wl.B<<' '<<wl.H<<std::endl;
	assert(util <= 1 + 1e-6);
	double tot_util = wl.tot_op * 1.0 / (tot_time * core.mac_num);
	MapCost cost(tot_energy, tot_time);
	// std::cout << "cost.time = " << cost.time << "ul2_ar.energy" << ul2_ar/cost.energy << "tot_util=" << tot_util << "\n";
	if(cost.cost() < best_map.cost.cost()){
		// Use current map instead.
		best_map = {cost, ubuf_energy, buf_energy, noc_energy, mac_energy, util, tot_util};
	}
}




///////////////////////////////////////////////////////////// Addiing Timeloop simulation ///////////////////////////////////////////////////////////
void EyerissInst::write_TL_hyperparams(const ConvWl& wl, const std::string working_path){
	
	// Prior work suggest 100% utilization rate of PE
	// Therefore, we obtain a more realistic utilization rate by timeloop
	// Provide the following information to the python script:

	// step 1: write param to .txt file
	std::ofstream param_file;
	param_file.open(working_path+"/prob_data.txt");
	// -- stream out all wl parameters using space as delimiter
	param_file << wl.B <<  " " << wl.W <<  " " << wl.H << " " 
	           << wl.C <<  " " << wl.K <<  " " 
			   << wl.S <<  " " << wl.R <<  " " 
			   << wl.sW << " " << wl.sH << "\n";
	param_file.close();

	std::ofstream arch_file;
	arch_file.open(working_path+"/arch_data.txt");
	// -- stream out all core parameters using space as delimiter
	arch_file << core.ul2.Size << " " << core.pes.Xarray << " " << core.pes.Yarray << "\n";
	arch_file.close();
}


CoreMapper::CoreMapping EyerissInst::read_TL_cost(const CoreMapper::ConvWl& wl, const std::string working_path){
	std::ifstream in(working_path+"/costs.txt");
	// # tot_util, E_weights, E_ifmap, E_psum, E_glb
	double tot_time;
	double E_wl1, E_al1, E_pl1, E_ubuf;
    in >> tot_time >> E_wl1 >> E_al1 >> E_pl1 >> E_ubuf;
	
	energy_t ubuf_energy=0, buf_energy=0, noc_energy=0, mac_energy=0;
	int layout_bias = 0*5;
	ubuf_energy = E_ubuf * wl.tot_op / 1000;
	buf_energy  = (E_wl1 + E_al1 + E_pl1) * wl.tot_op / 1000;
	noc_energy  = 0;
	mac_energy  = 0.173 * wl.tot_op;

	energy_t tot_energy = ubuf_energy + buf_energy + noc_energy + mac_energy;

	double util = wl.tot_op / (tot_time * core.pes.Xarray*core.pes.Yarray);
	double tot_util = util;
	
	MapCost cost(tot_energy, tot_time);
	best_map = {cost, ubuf_energy, buf_energy, noc_energy, mac_energy, util, tot_util};
	return best_map;
}
///////////////////////////////////////////////////////////// Addiing Timeloop simulation ///////////////////////////////////////////////////////////




///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////
std::vector<float> EyerissInst::gen_predictor_input(const ConvWl& wl, const Core& c) {
	
	float B, W, H, C_I, K_W, K_H, C_O;
    float meshX, meshY, Ubuf_width, Ubuf_size;
	float ifm_size, ofm_size, wgh_size, PE_size, H_x_res, CO_x_res, CO_xy_res, KH_y_res, CI_y_res, CO_y_res;

	B   = wl.B;
	W   = wl.W;
	H   = wl.H;
	C_I = wl.C;
	K_W = wl.S;
	K_H = wl.R;
	C_O = wl.K;

	meshX = core.pes.Xarray;
	meshY = core.pes.Yarray;
	Ubuf_size  = core.ul2.Size;

	ifm_size = B * W * H * C_I;
    ofm_size = B * W * H * C_O;
    wgh_size = C_O * C_I * K_W * K_H;
    PE_size   = meshX*meshY;
    H_x_res   = H   / meshX;
    CO_x_res  = C_O / meshX;
    CO_xy_res = C_O / PE_size;
    KH_y_res  = K_H / meshY;
    CI_y_res  = C_I / meshY;
    CO_y_res  = C_O / meshY;

	std::vector<float> input = {
        B, W, H, C_I, C_O, 
        meshX, meshY, Ubuf_size,
		ifm_size, ofm_size, wgh_size, log2(meshX), log2(meshY)};

	auto find_util = [](float data, std::vector<float>& input_o) {
		float result;
		for (int i = 0; i <= 6; i++) {
			result = data / static_cast<float>( ((static_cast<int>(data) + ((1 << i) - 1)) >> i) << i );
			input_o.push_back(result);
		}
	};
	
	find_util(B,   input);
	find_util(H,   input);
	find_util(C_O, input);
	find_util(K_H, input);
	find_util(C_I, input);

	return input;
}
///////////////////////////////////////////////////////////// Addiing performance predictors /////////////////////////////////////////////////////////////



// Codes for CoreMapper::ConvWl

CoreMapper::ConvWl::ConvWl(const CoreMapper::ConvParent& parent, len_t _B)
	:ConvParent (parent), B(_B), nGroup(1){}

void CoreMapper::ConvWl::init(){
	ConvParent::init();
	tot_op *= B;
}

vol_t CoreMapper::ConvWl::ifm_size() const{
	return ConvParent::ifm_size(B);
}

vol_t CoreMapper::ConvWl::ofm_size() const{
	return ConvParent::ofm_size(B);
}

void CoreMapper::ConvWl::calc_op(){
	ConvParent::update_op();
	tot_op *= B;
}

// Codes for CoreMapper::MapCost

CoreMapper::MapCost::MapCost(energy_t _energy, cycle_t _time)
	:energy(_energy),time(_time){}

bool CoreMapper::MapCost::is_valid() const{
	return energy < energy_inf;
}

cost_t CoreMapper::MapCost::cost(len_t nbatch) const{
	return calc_cost(energy, time*nbatch);
}

// Codes for CoreMapper::CoreMapping

CoreMapper::CoreMapping& CoreMapper::CoreMapping::operator*=(len_t factor){
	if(factor != 1){
		if(cost.energy != energy_inf) cost.energy *= factor;
		cost.time *= factor;
		ubuf *= factor;
		buffer *= factor;
		noc *= factor;
		mac *= factor;
	}
	return *this;
}

CoreMapper::CoreMapping& CoreMapper::CoreMapping::operator+=(const CoreMapping& other){
	if(cost.energy == energy_inf || other.cost.energy == energy_inf){
		cost.energy = energy_inf;
		return *this;
	}
	cost.energy += other.cost.energy;
	cost.time += other.cost.time;
	ubuf += other.ubuf;
	buffer += other.buffer;
	noc += other.noc;
	mac += other.mac;
	return *this;
}

// Codes for PolarInst::Part

std::string PolarInst::Part::getName() const{
	return std::string(1,partName[static_cast<std::uint8_t>(dimA)])+partName[static_cast<std::uint8_t>(dimO)];
}

/*
void PolarMapper::_try_print(const CoreMapper::Workload& wl, size_t i){
	static std::ofstream ofs("out.log");
	ofs<<wl.C<<'\t'<<wl.K<<'\t'<<wl.H<<'\t'<<wl.W<<'\t';
	ofs<<wl.R<<'\t'<<wl.S<<'\t'<<wl.B<<'\t';
	ofs<<all_parts[i].getName()<<'\t';
	ofs<<H1.cnt<<'\t'<<W1.cnt<<'\t';
	//ofs<<hasWL2<<'\t'<<hasKBSD<<'\t'<<hasWBSD<<'\t';
	ofs<<allcst.time<<'\t'<<allcst.energy<<'\t';
	ofs<<hasl2.time<<'\t'<<hasl2.energy<<'\t';
	ofs<<comp_time<<std::endl;
}
*/

#undef PolarInst
#undef EyerissInst
