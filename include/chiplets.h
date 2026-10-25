#ifndef CHIPLETS_H
#define CHIPLETS_H

// chess H3: heterogeneous packages.
//
// A package is an x_cut x y_cut grid of chiplets; every chiplet holds x_step x y_step
// cores of one chiplet type. A chiplet type is a core micro-architecture (mm), a MAC
// count, an L2 (ul3) size and a technology (its node): it owns the Core the mapper
// evaluates and the CoreMapper itself, and knows its die area. Upstream Gemini is
// the special case of one type on every chiplet (ChipletMap::homogeneous), for which
// every number this file computes is the one main.cpp computed before.
//
// A chiplet map file (the optional 21st stdin token, `@<path>`):
//
//   types 2
//   type 0 mm 0 mac 512 ul3_kb 1024 tech technology.tech
//   type 1 mm 0 mac 1024 ul3_kb 512 tech technology.7nm.tech
//   map 0 1          # one line per chiplet row y = 0 .. y_cut-1, x_cut entries each
//   map 1 0
//
// Package-level terms (DRAM, NoP PHY, SerDes, packaging) come from the base technology
// of the first stdin token; a type's technology supplies the node terms (MAC and buffer
// energies, densities, area factors, NoC hop energy). The package runs at one clock, the
// slowest type's; the NoC hop energy is the core-count-weighted mean of the types' values.

#include <memory>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "cluster.h"
#include "core.h"
#include "coremapping.h"
#include "technology.h"
#include "util.h"

// Upstream's SRAM access-energy table (per byte: [0] read, [1] write), moved from main.cpp.
std::vector<double> buffer(int width, vol_t size);

struct ChipletType {
	int idx = 0;
	int mm = 0;                   // 0: PolarCore (NVDLA style), 1: EyerissCore
	std::uint16_t mac_dim = 512;  // Polar: MACs per core (512, 1024, 2048, 4096, 8192); Eyeriss: array side, as upstream
	vol_t ul3 = 0;                // L2 bytes
	Technology tech;              // node terms
	std::uint16_t vector_len = 0, lane_len = 0;
	Core::numMac_t LR_mac_num = 0;
	std::unique_ptr<Core> core;
	std::unique_ptr<CoreMapper> mapper;
	cidx_t num_chiplets = 0;      // chiplets of this type in the package
	double core_area_um2 = 0;     // post-layout core area
	double compute_die_area_mm2 = 0;  // one chiplet of this type (with D2D PHY, control and DFT)

	// Core and mapper from mm, mac_dim, ul3 and tech, with upstream's buffer sizes and energies.
	void build();
	// Peak MACs per cycle of one core (upstream's 16 * vector_len * lane_len).
	unsigned macs_per_core() const;
};

class ChipletMap {
public:
	std::vector<ChipletType> types;
	std::vector<int> grid;  // grid[cy * x_cut + cx] = type index
	mlen_t x_cut = 1, y_cut = 1, x_step = 1, y_step = 1;

	// One type on every chiplet: upstream's package.
	static ChipletMap homogeneous(int mm, std::uint16_t mac_dim, vol_t ul3, const Technology& T);
	// The file form above. Types without a `tech` entry take `base`.
	static ChipletMap load(const std::string& path, const Technology& base);

	bool single() const { return types.size() == 1; }
	int type_at(mlen_t cx, mlen_t cy) const;
	int type_of_pos(pos_t p) const;
	int type_of_core(cidx_t core) const;
	const ChipletType& type(int idx) const { return types[idx]; }

	// The chiplet types a cluster's cores belong to, with core counts.
	struct Mix {
		std::vector<std::pair<int, cidx_t>> counts;
		cidx_t total = 0;
		bool single() const { return counts.size() == 1; }
	};
	Mix mix(const Cluster& c) const;

	// What a layer engine needs of a cluster: the L2 every core can hold (the smallest), its access
	// energies (core-count weighted), and the mappers to evaluate the layer with.
	struct ClusterView {
		Core::Buffer ubuf;
		std::vector<std::pair<const CoreMapper*, cidx_t>> mappers;
		cidx_t total = 0;
		// Energy per core (count-weighted mean) and the slowest core's time; invalid if any type cannot map the layer.
		CoreMapper::CoreMapping genLayerMap(const Layer& layer, const PartSch& part, const FetchSch& fetch, len_t batch_size, bool wgtB) const;
	};
	ClusterView view(const Cluster& c) const;

	vol_t min_ubuf(const Cluster& c) const;  // smallest L2 among the cluster's core types
	vol_t min_ubuf() const;                  // over the package
	vol_t min_core_ubuf() const;             // smallest core buffer the mappers see (Eyeriss: its fixed ul2)
	energy_t noc_hop_cost() const;           // core-count weighted mean over the package (the type's value when single)
	double frequency_ghz() const;            // the package clock: the slowest type's (the type's own when single)
	double tops() const;                     // upstream's peak formula summed over chiplets
	cidx_t cores_per_chiplet() const { return static_cast<cidx_t>(x_step) * y_step; }
	cidx_t num_chiplets() const { return static_cast<cidx_t>(x_cut) * y_cut; }

	// Die areas per type from the package geometry and the base technology's NoP PHY (upstream's formulas).
	void compute_areas(const Technology& base, bw_t NoC_bw, bw_t NoP_bw);
	// Compute die area as the result line reports it: the type's when single, else the mean over chiplets.
	double compute_die_area_mm2() const;
	double total_compute_die_area_mm2() const;
	double max_compute_die_area_mm2() const;

	void dump(std::ostream& os) const;
};

#endif // CHIPLETS_H
