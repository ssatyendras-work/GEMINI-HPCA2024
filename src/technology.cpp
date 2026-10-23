// Technology parameters of a run (chess/main, T3). See include/technology.h.
#include "technology.h"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

Technology Technology::builtin(const std::string& tech, const std::string& package_type, const std::string& IO_type, const std::string& DDR_type) {
	// The branches of upstream main.cpp (ea8700a), moved here unchanged.
	Technology T{};
	T.node = tech;
	if (tech == "7") {
		T.scale_factor = 0.7;
		T.power_factor = 0.7;
		T.frequency_ghz = 1.3;
	} else if (tech == "12") {
		T.scale_factor = 1;
		T.power_factor = 1;
		T.frequency_ghz = 1;
	} else {
		throw std::runtime_error("Technology not supported: " + tech);
	}
	T.sram_den = 0.0964 * 8 * T.scale_factor;
	T.mac_den = 57.9 * T.scale_factor;
	T.noc_den = 16781.312 * T.scale_factor;
	T.nop_den = 0;
	T.ddr_phy_den = 0;
	T.ddr_ctrl_den = 0;
	T.ddr_cost = 0;
	if (DDR_type == "LPDDR5") {  // not use
		T.dram_acc_cost = 0;
	} else if (DDR_type == "GDDR6X") {
		T.ddr_phy_den = 5940 * 1100 / (14 / 8 * 32);  // integer division, as upstream
		T.ddr_ctrl_den = 0 * 0.09 * T.scale_factor / (14 / 8 * 32);
		T.ddr_cost = 3.5 / (14 / 8 * 32);
		T.dram_acc_cost = 10.5 * 8;
	} else if (DDR_type == "HBM") {  // not use
		T.ddr_phy_den = 502665 / 819;
		T.ddr_ctrl_den = 0 * T.scale_factor;
		T.ddr_cost = 320 / 819;
		T.dram_acc_cost = 5.5 * 8;
	} else {
		throw std::runtime_error("DDR type not supported.");
	}
	T.turnover_factor = 0.3 / 0.5;
	T.noc_hop_cost = 0.4 * 8 * T.power_factor;
	T.lr_mac_cost = 0.0873 * T.power_factor;
	T.mac_cost = 0.018;
	T.serdes = false;
	T.serdes_power_mw = 0;
	T.serdes_lane_gbps = 0;
	T.nop_hop_cost = 0;
	if (IO_type == "XSR") {
		T.serdes = true;
		T.serdes_lane_gbps = 112;
		if (tech == "12") {
			T.serdes_power_mw = 97.44;
			T.nop_den = 960.816 * 1339.008 / 112;
		} else {
			T.serdes_power_mw = 97.44 * 0.5 / 0.7;
			T.nop_den = 1020 * 1060 / 112;
		}
	} else if (IO_type == "USR") {
		if (tech == "12") {
			T.nop_den = 960.816 * 1339.008 / 56;
			T.nop_hop_cost = 2.4 * 8;
		} else {
			T.nop_den = 1020 * 1060 / 56;
			T.nop_hop_cost = 1.7 * 8;
		}
	} else if (IO_type == "UCIe") {
		if (package_type == "SI") {
			T.nop_den = 1000 * 1000 / 947;
			T.nop_hop_cost = 0.3 * 8;
		} else if (package_type == "FO") {
			T.nop_den = 1000 * 1000 / 631;
			T.nop_hop_cost = 0.6 * 8;
		} else if (package_type == "OS") {
			T.nop_den = 1000 * 1000 / 145;
			T.nop_hop_cost = 1 * 8;
		}
	} else {
		throw std::runtime_error("IO type not supported.");
	}
	T.post_layout_scale = 2.2;
	T.control_unit_prop = 1.05;
	T.dft_prop = 1.05;
	T.max_die_area_mm2 = 858;
	T.ol2_read = 0.07648125 * 8 * T.power_factor;
	T.ol2_write = 0.0989875 * 8 * T.power_factor;
	T.ul3_read = 0.217125 * 8 * T.power_factor;
	T.ul3_write = 0.234025 * 8 * T.power_factor;
	return T;
}

std::vector<std::pair<std::string, double>> Technology::numbers() const {
	return {
		{"frequency_ghz", frequency_ghz}, {"scale_factor", scale_factor}, {"power_factor", power_factor},
		{"sram_den", sram_den}, {"mac_den", mac_den}, {"noc_den", noc_den}, {"nop_den", nop_den}, {"ddr_phy_den", ddr_phy_den}, {"ddr_ctrl_den", ddr_ctrl_den},
		{"post_layout_scale", post_layout_scale}, {"control_unit_prop", control_unit_prop}, {"dft_prop", dft_prop}, {"max_die_area_mm2", max_die_area_mm2},
		{"mac_cost", mac_cost}, {"lr_mac_cost", lr_mac_cost}, {"noc_hop_cost", noc_hop_cost}, {"nop_hop_cost", nop_hop_cost}, {"dram_acc_cost", dram_acc_cost},
		{"turnover_factor", turnover_factor}, {"ul3_read", ul3_read}, {"ul3_write", ul3_write}, {"ol2_read", ol2_read}, {"ol2_write", ol2_write},
		{"serdes", serdes ? 1.0 : 0.0}, {"serdes_power_mw", serdes_power_mw}, {"serdes_lane_gbps", serdes_lane_gbps}, {"ddr_cost", ddr_cost},
	};
}

void Technology::dump(std::ostream& out) const {
	out.precision(17);
	out << "node " << node << "\n";
	for (const auto& kv : numbers()) out << kv.first << " " << kv.second << "\n";
}

Technology Technology::load(const std::string& path) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open technology file " + path);
	std::map<std::string, std::string> kv;
	std::string line;
	while (std::getline(in, line)) {
		auto hash = line.find('#');
		if (hash != std::string::npos) line = line.substr(0, hash);
		std::istringstream s(line);
		std::string key, value;
		if (s >> key >> value) kv[key] = value;
	}
	Technology T{};
	auto take = [&](const std::string& key) {
		auto it = kv.find(key);
		if (it == kv.end()) throw std::runtime_error("technology file " + path + " lacks " + key);
		return std::stod(it->second);
	};
	if (!kv.count("node")) throw std::runtime_error("technology file " + path + " lacks node");
	T.node = kv["node"];
	T.frequency_ghz = take("frequency_ghz");
	T.scale_factor = take("scale_factor");
	T.power_factor = take("power_factor");
	T.sram_den = take("sram_den");
	T.mac_den = take("mac_den");
	T.noc_den = take("noc_den");
	T.nop_den = take("nop_den");
	T.ddr_phy_den = take("ddr_phy_den");
	T.ddr_ctrl_den = take("ddr_ctrl_den");
	T.post_layout_scale = take("post_layout_scale");
	T.control_unit_prop = take("control_unit_prop");
	T.dft_prop = take("dft_prop");
	T.max_die_area_mm2 = take("max_die_area_mm2");
	T.mac_cost = take("mac_cost");
	T.lr_mac_cost = take("lr_mac_cost");
	T.noc_hop_cost = take("noc_hop_cost");
	T.nop_hop_cost = take("nop_hop_cost");
	T.dram_acc_cost = take("dram_acc_cost");
	T.turnover_factor = take("turnover_factor");
	T.ul3_read = take("ul3_read");
	T.ul3_write = take("ul3_write");
	T.ol2_read = take("ol2_read");
	T.ol2_write = take("ol2_write");
	T.serdes = take("serdes") != 0;
	T.serdes_power_mw = take("serdes_power_mw");
	T.serdes_lane_gbps = take("serdes_lane_gbps");
	T.ddr_cost = take("ddr_cost");
	return T;
}
