// Technology parameters of a run (chess/main, T3).
//
// Every number main.cpp used to choose from its `tech`, `package_type`, `IO_type`
// and `DDR_type` branches lives here. `builtin` reproduces those branches exactly;
// `load` reads the same fields from a file written by chess from its technology
// descriptors ("key value" lines, '#' comments). The stdin token `tech` selects:
// "7" or "12" as before, or "@<path>" to load a file. All energies are the final
// values main.cpp uses (power factor applied); `power_factor` itself remains for
// the buffer table and the Eyeriss buffer constants that stay in main.cpp.
#ifndef TECHNOLOGY_H
#define TECHNOLOGY_H

#include <string>
#include <vector>

struct Technology {
	std::string node;          // cost.cpp key: "7", "12", ...
	double frequency_ghz;      // SchNode::frequency
	double scale_factor;       // area scale vs 12 nm
	double power_factor;       // energy scale vs 12 nm (buffer table, Eyeriss buffers)
	// areas (um2)
	double sram_den;           // per byte
	double mac_den;            // per 8-bit MAC
	double noc_den;            // per unit of NoC bandwidth
	double nop_den;            // per unit of NoP bandwidth (D2D PHY)
	double ddr_phy_den;        // per unit of DRAM bandwidth
	double ddr_ctrl_den;       // per unit of DRAM bandwidth
	double post_layout_scale;  // core area multiplier
	double control_unit_prop;  // die area multiplier
	double dft_prop;           // die area multiplier
	double max_die_area_mm2;   // reticle limit
	// energies (pJ)
	double mac_cost;           // per 8-bit MAC (PE setting and buses)
	double lr_mac_cost;        // per FP16 MAC
	double noc_hop_cost;       // per byte-hop on the NoC
	double nop_hop_cost;       // per byte-hop between dies (0 when a SerDes power model applies)
	double dram_acc_cost;      // per byte of DRAM access
	double turnover_factor;    // switching activity applied to buffer accesses
	double ul3_read, ul3_write, ol2_read, ol2_write;  // per byte at 1 MiB, before the activity factor, power factor applied
	// SerDes (XSR) power model
	bool serdes;
	double serdes_power_mw;    // per lane
	double serdes_lane_gbps;   // lane rate
	// DRAM cost
	double ddr_cost;           // USD per unit of DRAM bandwidth (HBM: per GB, as upstream)

	static Technology builtin(const std::string& tech, const std::string& package_type, const std::string& IO_type, const std::string& DDR_type);
	static Technology load(const std::string& path);
	std::vector<std::pair<std::string, double>> numbers() const;  // every numeric field by key, in file order
	void dump(std::ostream& out) const;                            // the file form
};

#endif
