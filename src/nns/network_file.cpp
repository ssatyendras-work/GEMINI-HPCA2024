// chess/main (E8): a network read from a text file, so chess's own model graphs reach the search without a rebuild.
//
// The file is written by chess (chess.providers.gemini.networks.to_file), one layer per line in network order; '#' starts a comment:
//
//   network <name>
//   <Type> <layer name> <field>=<value> ... [prevs=<i>,<j>] [wgts=<i>,<j>] [ext=<name>;<c>;<h>;<w>|...] [bits=<n>]
//
// Type and fields are those of the layer classes in layer.h, as the networks in src/nns write them with NLAYER:
//   Conv (C K R S H W sH sW), GroupConv (G C K R S H W sH sW), FC (C K IH IW), Pooling (K H W R S sH sW), Eltwise (N K H W),
//   PTP (K H W), Transpose (K H W order=<c|h|w>,<c|h|w>,<c|h|w>: the source dimension of the output's C, H and W).
// Every field of a type is required, so no layer.h default is taken silently. prevs and wgts name earlier layers by index (Network::add's
// ifmap and weight predecessors), ext the external inputs with their shapes (';' and '|' never occur in chess's ids), bits the layer's
// bitwidth (0: the network's default).

#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <stdexcept>
#include "layer.h"
#include "nns/nns.h"

namespace {

[[noreturn]] void fail(const std::string& path, int line, const std::string& what) {
	throw std::runtime_error("network file " + path + ":" + std::to_string(line) + ": " + what);
}

std::vector<std::string> split(const std::string& s, char sep) {
	std::vector<std::string> out;
	if (s.empty()) return out;
	std::string part;
	std::istringstream in(s);
	while (std::getline(in, part, sep)) out.push_back(part);
	return out;
}

}  // namespace

Network load_network(const std::string& path, std::string& name) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open network file " + path);
	Network n;
	name.clear();
	Network::lid_t count = 0;
	std::string line;
	int lineno = 0;
	while (std::getline(in, line)) {
		++lineno;
		auto hash = line.find('#');
		if (hash != std::string::npos) line = line.substr(0, hash);
		std::istringstream s(line);
		std::string type, lname, tok;
		if (!(s >> type)) continue;
		if (type == "network") {
			if (!(s >> name)) fail(path, lineno, "network needs a name");
			continue;
		}
		if (!(s >> lname)) fail(path, lineno, type + " without a layer name");
		std::map<std::string, std::string> kv;
		while (s >> tok) {
			auto eq = tok.find('=');
			if (eq == std::string::npos || eq == 0) fail(path, lineno, "expected <field>=<value>, got " + tok);
			if (!kv.emplace(tok.substr(0, eq), tok.substr(eq + 1)).second) fail(path, lineno, "field " + tok.substr(0, eq) + " given twice");
		}
		std::set<std::string> used;
		auto integer = [&](const std::string& key, const std::string& text) -> long {
			size_t pos = 0;
			long v = -1;
			try { v = std::stol(text, &pos); } catch (const std::exception&) { pos = 0; }
			if (pos != text.size() || v < 0) fail(path, lineno, lname + ": " + key + " is not a non-negative integer (" + text + ")");
			return v;
		};
		auto num = [&](const std::string& key) -> len_t {
			auto it = kv.find(key);
			if (it == kv.end()) fail(path, lineno, type + " " + lname + " lacks " + key);
			used.insert(key);
			return static_cast<len_t>(integer(key, it->second));
		};
		auto layers = [&](const std::string& key) {
			Network::layer_set out;
			auto it = kv.find(key);
			if (it == kv.end()) return out;
			used.insert(key);
			for (const auto& x : split(it->second, ',')) {
				long v = integer(key, x);
				if (v >= count) fail(path, lineno, lname + ": " + key + " names layer " + x + ", which is not an earlier one");
				out.push_back(static_cast<Network::lid_t>(v));
			}
			return out;
		};
		std::vector<InputData> ext;
		if (kv.count("ext")) {
			used.insert("ext");
			for (const auto& e : split(kv["ext"], '|')) {
				auto f = split(e, ';');
				if (f.size() != 4 || f[0].empty()) fail(path, lineno, lname + ": an external input is <name>;<c>;<h>;<w>, got " + e);
				ext.emplace_back(f[0], fmap_shape(integer("ext", f[1]), integer("ext", f[2]), integer("ext", f[3])));
			}
		}
		Network::layer_set prevs = layers("prevs"), wgts = layers("wgts");
		bwidth_t bits = kv.count("bits") ? static_cast<bwidth_t>(num("bits")) : 0;
		std::unique_ptr<const Layer> l;
		if (type == "Conv") {
			ConvLayer::Workload w;
			w.C = num("C"); w.K = num("K"); w.R = num("R"); w.S = num("S"); w.H = num("H"); w.W = num("W"); w.sH = num("sH"); w.sW = num("sW");
			l.reset(new ConvLayer(lname, w));
		} else if (type == "GroupConv") {
			GroupConvLayer::Workload w;
			w.G = num("G"); w.C = num("C"); w.K = num("K"); w.R = num("R"); w.S = num("S"); w.H = num("H"); w.W = num("W"); w.sH = num("sH"); w.sW = num("sW");
			l.reset(new GroupConvLayer(lname, w));
		} else if (type == "FC") {
			FCLayer::Workload w;
			w.C = num("C"); w.K = num("K"); w.IH = num("IH"); w.IW = num("IW");
			l.reset(new FCLayer(lname, w));
		} else if (type == "Pooling") {
			PoolingLayer::Workload w;
			w.K = num("K"); w.H = num("H"); w.W = num("W"); w.R = num("R"); w.S = num("S"); w.sH = num("sH"); w.sW = num("sW");
			l.reset(new PoolingLayer(lname, w));
		} else if (type == "Eltwise") {
			EltwiseLayer::Workload w;
			w.N = num("N"); w.K = num("K"); w.H = num("H"); w.W = num("W");
			l.reset(new EltwiseLayer(lname, w));
		} else if (type == "PTP") {
			PTPLayer::Workload w;
			w.K = num("K"); w.H = num("H"); w.W = num("W");
			l.reset(new PTPLayer(lname, w));
		} else if (type == "Transpose") {
			TransposeLayer::Workload w;
			w.K = num("K"); w.H = num("H"); w.W = num("W");
			if (!kv.count("order")) fail(path, lineno, type + " " + lname + " lacks order");
			used.insert("order");
			auto order = split(kv["order"], ',');
			const std::map<std::string, TransposeLayer::dim> dims{{"c", TransposeLayer::dim::C}, {"h", TransposeLayer::dim::H}, {"w", TransposeLayer::dim::W}};
			if (order.size() != 3) fail(path, lineno, lname + ": order names three dimensions, got " + kv["order"]);
			for (int i = 0; i < 3; ++i) {
				auto d = dims.find(order[i]);
				if (d == dims.end()) fail(path, lineno, lname + ": order takes c, h and w, got " + order[i]);
				w.order[i] = d->second;
			}
			l.reset(new TransposeLayer(lname, w));
		} else {
			fail(path, lineno, "unknown layer type " + type + " (Conv, GroupConv, FC, Pooling, Eltwise, PTP, Transpose)");
		}
		for (const auto& [k, v] : kv)
			if (!used.count(k)) fail(path, lineno, lname + ": " + type + " has no field " + k);
		n.add(l.release(), prevs, bits, ext, wgts);
		++count;
	}
	if (count == 0) throw std::runtime_error("network file " + path + " has no layers");
	if (name.empty()) name = path;
	return n;
}
