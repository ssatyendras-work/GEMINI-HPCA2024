// chess H3: heterogeneous packages (see include/chiplets.h).

#include "chiplets.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "network.h"
#include "noc.h"

#define KB *1024

// ---------------------------------------------------------------------------------------------- buffer table

std::vector<double> buffer(int width, vol_t size) {
	std::vector<double> access;//per byte 0 is read, 1 is write
	access.resize(2);
	vol_t size_;
	if (width >= 512) {
		size_ = size / (width / 256);
		access = buffer(256, size_);
		access[0] *= sqrt(size / size_);
		access[1] *= sqrt(size / size_);
		return access;
	}
	if (width == 256) {
		if (size == 32 KB) {
			access[0] = 0.065678125;
			access[1] = 0.0641375;
		}
		else if (size == 16 KB) {
			access[0] = 0.06311875;
			access[1] = 0.05563125;
		}
		else if (size == 8 KB) {
			access[0] = 0.051290625;
			access[1] = 0.04560625 ;
		}
		else if (size == 4 KB) {
			access[0] = 0.049;
			access[1] = 0.064;
		}
		else {
			throw std::runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 128) {
		if (size == 32 KB) {
			access[0] = 0.106675;
			access[1] = 0.10539375;
		}
		else if (size == 16 KB) {
			access[0] = 0.06848125;
			access[1] = 0.06684375;
		}
		else if (size == 8 KB) {
			access[0] = 0.06553125;
			access[1] = 0.057975;
		}
		else if (size == 4 KB) {
			access[0] = 0.05336875;
			access[1] = 0.0476625;
		}
		else if (size == 2 KB) {
			access[0] = 0.06025625;
			access[1] = 0.06025625;
		}
		else {
			throw std::runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 64) {
		if (size == 32 KB) {
			access[0] = 0.194775;
			access[1] = 0.1923;
		}
		else if (size == 16 KB) {
			access[0] = 0.112;
			access[1] = 0.110675;
		}
		else if (size == 8 KB) {
			access[0] = 0.0740875;
			access[1] = 0.0722625;
		}
		else if (size == 4 KB||size == 4.5 KB) {
			access[0] = 0.0703625;
			access[1] = 0.0626625;
		}
		else if (size == 2 KB) {
			access[0] = 0.057525;
			access[1] = 0.051775;
		}
		else {
			throw std::runtime_error("Cannot find buffer energy.");
		}
	}
	if (width == 32) {
		if (size == 32 KB) {
			access[0] = 0.194775;
			access[1] = 0.1923;
		}
		else if (size == 16 KB) {
			access[0] = 0.112;
			access[1] = 0.110675;
		}
		else if (size == 8 KB) {
			access[0] = 0.11608;
			access[1] = 0.13898;
		}
		else if (size == 4 KB) {
			access[0] = 0.07422;
			access[1] = 0.09414;
		}
		else if (size == 2 KB) {
			access[0] = 0.06326;
			access[1] = 0.08034;
		}
		else if (size == 1 KB) {
			access[0] = 0.07952;
			access[1] = 0.0981;
		}
		else {
			throw std::runtime_error("Cannot find buffer energy.");
		}
	}
	return access;
}

// ---------------------------------------------------------------------------------------------- ChipletType

static void mac_geometry(std::uint16_t mac_dim, std::uint16_t& vector_len, std::uint16_t& lane_len) {
	if (mac_dim == 512) {
		vector_len = 4;
		lane_len = 8;
	}
	else if (mac_dim == 1024) {
		vector_len = 8;
		lane_len = 8;
	}
	else if (mac_dim == 2048) {
		vector_len = 8;
		lane_len = 16;
	}
	else if (mac_dim == 4096) {
		vector_len = 16;
		lane_len = 16;
	}
	else if (mac_dim == 8192) {
		vector_len = 16;
		lane_len = 32;
	}
	else {
		throw std::runtime_error("MAC scale not supported·");
	}
}

unsigned ChipletType::macs_per_core() const {
	return 16u * vector_len * lane_len;
}

void ChipletType::build() {
	mac_geometry(mac_dim, vector_len, lane_len);
	const Technology& T = tech;
	double power_factor = T.power_factor;
	double turnover_factor = T.turnover_factor;
	energy_t LR_mac_cost = T.lr_mac_cost; //IEEE FP16
	LR_mac_num = vector_len * lane_len * 16 / 16;
	vol_t ul3_ = ul3;
	if (mm == 0) {
		PolarCore::Buffer al1, wl1, ol1, al2, wl2, ol2, ul3b;
		PolarCore::PESetting s(vector_len, lane_len, T.mac_cost);
		PolarCore::Bus bus(4, 4, T.mac_cost, 64);
		al1.Size = 8 * vector_len / 8 KB;
		ol1.Size = 2 * lane_len / 8 KB;
		wl1.Size = 4 * lane_len*vector_len / 64 KB;
		ol2.Size = 28 * vector_len * lane_len / 64 KB;
		wl2.Size = 0;//256 KB;
		ul3b.Size = ul3_; //16 64-bit IO 64KB 1-port MBSRAM
		al2.Size = 0;
		al1.RCost = (buffer(vector_len * 8, al1.Size)[0]+0.1*lane_len/8) * 8 * turnover_factor *power_factor;
		al1.WCost = buffer(vector_len * 8, al1.Size)[1] * 8 * turnover_factor * power_factor;
		wl1.RCost = buffer(vector_len * 8, wl1.Size)[0] * 8 * turnover_factor * power_factor;
		wl1.WCost = buffer(vector_len * 8, wl1.Size)[1] * 8 * turnover_factor * power_factor;
		ol1.RCost = buffer(lane_len * 16, ol1.Size)[0] * 8 * turnover_factor * power_factor;
		ol1.WCost = buffer(lane_len * 16, ol1.Size)[1] * 8 * turnover_factor * power_factor;
		ol2.RCost = T.ol2_read * ul3_ / (1024 KB) * turnover_factor;// ol2 is a small part of ul3
		ol2.WCost = T.ol2_write * ul3_ / (1024 KB) *turnover_factor;
		ul3b.RCost = T.ul3_read * ul3_ /(1024 KB) * turnover_factor;
		ul3b.WCost = T.ul3_write * ul3_ /(1024 KB)* turnover_factor;
		al2.RCost = al2.WCost = 0 * power_factor;
		wl2.RCost = wl2.WCost = 0 * power_factor;
		auto c = std::make_unique<PolarCore>(s, bus, al1, wl1, ol1, al2, wl2, ol2, ul3b, LR_mac_num, LR_mac_cost);
		mapper = std::make_unique<PolarMapper>(*c);
		core = std::move(c);
	} else {
		EyerissCore::Buffer _al1, _wl1, pl1, ul2;
		EyerissCore::PESetting s2(mac_dim, mac_dim, T.mac_cost);
		EyerissCore::Bus ibus(T.mac_cost, 64);
		EyerissCore::Bus wbus(T.mac_cost, 64);
		EyerissCore::Bus pbus(T.mac_cost, 64); // ifmap RC, weight RCK, psum RK
		_al1.Size = 32;
		pl1.Size = 1;
		_wl1.Size = 128;
		ul2.Size = 1024 KB;
		_al1.RCost = 0.0509 * 8 * turnover_factor *power_factor; //8bit IO single port
		_al1.WCost = 0.0506 * 8 * turnover_factor*power_factor;//0.045;
		_wl1.RCost = 0.0545 * 8 * turnover_factor*power_factor; //Using 2 banks of 64
		_wl1.WCost = 0.054 * 8 * turnover_factor*power_factor;//0.090;
		pl1.RCost = pl1.WCost = 0.0 * turnover_factor*power_factor;
		ul2.RCost = 0.1317125 * 8 * turnover_factor*power_factor;
		ul2.WCost = 0.234025 * 8 * turnover_factor*power_factor;
		auto c = std::make_unique<EyerissCore>(s2, ibus, wbus, pbus, _al1, _wl1, pl1, ul2, LR_mac_num, LR_mac_cost);
		mapper = std::make_unique<EyerissMapper>(*c);
		core = std::move(c);
	}
}

// ---------------------------------------------------------------------------------------------- ChipletMap

static void geometry(ChipletMap& m) {
	m.x_cut = NoC::x_cut;
	m.y_cut = NoC::y_cut;
	m.x_step = NoC::x_step;
	m.y_step = NoC::y_step;
}

ChipletMap ChipletMap::homogeneous(int mm, std::uint16_t mac_dim, vol_t ul3, const Technology& T) {
	ChipletMap m;
	geometry(m);
	m.types.resize(1);
	ChipletType& t = m.types[0];
	t.idx = 0;
	t.mm = mm;
	t.mac_dim = mac_dim;
	t.ul3 = ul3;
	t.tech = T;
	t.num_chiplets = m.num_chiplets();
	t.build();
	m.grid.assign(static_cast<size_t>(m.x_cut) * m.y_cut, 0);
	return m;
}

ChipletMap ChipletMap::load(const std::string& path, const Technology& base) {
	std::ifstream in(path);
	if (!in) throw std::runtime_error("cannot open chiplet map " + path);
	ChipletMap m;
	geometry(m);
	int ntypes = -1;
	std::vector<std::vector<int>> rows;
	std::string line;
	while (std::getline(in, line)) {
		auto hash = line.find('#');
		if (hash != std::string::npos) line = line.substr(0, hash);
		std::istringstream s(line);
		std::string key;
		if (!(s >> key)) continue;
		if (key == "types") {
			s >> ntypes;
			if (ntypes < 1) throw std::runtime_error("chiplet map " + path + ": types must be >= 1");
			m.types.resize(ntypes);
		} else if (key == "type") {
			int idx;
			if (!(s >> idx) || ntypes < 0 || idx < 0 || idx >= ntypes) throw std::runtime_error("chiplet map " + path + ": bad type index in '" + line + "'");
			ChipletType& t = m.types[idx];
			t.idx = idx;
			t.tech = base;
			std::string k, v;
			bool has_mm = false, has_mac = false, has_ul3 = false;
			while (s >> k >> v) {
				if (k == "mm") { t.mm = std::stoi(v); has_mm = true; }
				else if (k == "mac") { t.mac_dim = static_cast<std::uint16_t>(std::stoi(v)); has_mac = true; }
				else if (k == "ul3_kb") { t.ul3 = static_cast<vol_t>(std::stoull(v)) KB; has_ul3 = true; }
				else if (k == "tech") { t.tech = Technology::load(v); }
				else throw std::runtime_error("chiplet map " + path + ": unknown key " + k);
			}
			if (!has_mm || !has_mac || !has_ul3) throw std::runtime_error("chiplet map " + path + ": type " + std::to_string(idx) + " needs mm, mac and ul3_kb");
		} else if (key == "map") {
			std::vector<int> row;
			int v;
			while (s >> v) row.push_back(v);
			rows.push_back(row);
		} else {
			throw std::runtime_error("chiplet map " + path + ": unknown line '" + line + "'");
		}
	}
	if (ntypes < 0) throw std::runtime_error("chiplet map " + path + ": no types line");
	if (static_cast<int>(rows.size()) != m.y_cut) throw std::runtime_error("chiplet map " + path + ": " + std::to_string(rows.size()) + " map rows for y_cut " + std::to_string(m.y_cut));
	m.grid.assign(static_cast<size_t>(m.x_cut) * m.y_cut, -1);
	for (int cy = 0; cy < m.y_cut; ++cy) {
		if (static_cast<int>(rows[cy].size()) != m.x_cut) throw std::runtime_error("chiplet map " + path + ": map row " + std::to_string(cy) + " has " + std::to_string(rows[cy].size()) + " entries for x_cut " + std::to_string(m.x_cut));
		for (int cx = 0; cx < m.x_cut; ++cx) {
			int v = rows[cy][cx];
			if (v < 0 || v >= ntypes) throw std::runtime_error("chiplet map " + path + ": type " + std::to_string(v) + " out of range");
			m.grid[cy * m.x_cut + cx] = v;
			m.types[v].num_chiplets++;
		}
	}
	for (ChipletType& t : m.types) {
		// a type's own clock is not used: the package runs at the base technology's clock (SchNode::frequency)
		t.build();
	}
	return m;
}

int ChipletMap::type_at(mlen_t cx, mlen_t cy) const {
	return grid[static_cast<size_t>(cy) * x_cut + cx];
}

int ChipletMap::type_of_pos(pos_t p) const {
	return type_at(static_cast<mlen_t>(p.x / x_step), static_cast<mlen_t>(p.y / y_step));
}

int ChipletMap::type_of_core(cidx_t core) const {
	return type_of_pos(Cluster::get_pos(core));
}

ChipletMap::Mix ChipletMap::mix(const Cluster& c) const {
	Mix m;
	m.total = c.num_cores();
	if (single()) {
		m.counts.emplace_back(0, m.total);
		return m;
	}
	std::vector<cidx_t> cnt(types.size(), 0);
	for (cidx_t i = 0; i < m.total; ++i) {
		cnt[type_of_pos(c[i])]++;
	}
	for (size_t t = 0; t < cnt.size(); ++t) {
		if (cnt[t]) m.counts.emplace_back(static_cast<int>(t), cnt[t]);
	}
	return m;
}

ChipletMap::ClusterView ChipletMap::view(const Cluster& c) const {
	ClusterView v;
	Mix m = mix(c);
	v.total = m.total;
	if (m.single()) {
		const ChipletType& t = types[m.counts[0].first];
		v.ubuf = t.core->ubuf();
		v.mappers.emplace_back(t.mapper.get(), m.total);
		return v;
	}
	// The L2 every core can hold is the smallest; its access energies are the cores' mean.
	const Core::Buffer* smallest = nullptr;
	double rcost = 0, wcost = 0;
	for (const auto& [idx, n] : m.counts) {
		const ChipletType& t = types[idx];
		const Core::Buffer& b = t.core->ubuf();
		if (!smallest || b.Size < smallest->Size) smallest = &b;
		rcost += b.RCost * n;
		wcost += b.WCost * n;
		v.mappers.emplace_back(t.mapper.get(), n);
	}
	v.ubuf = *smallest;
	v.ubuf.RCost = rcost / m.total;
	v.ubuf.WCost = wcost / m.total;
	return v;
}

CoreMapper::CoreMapping ChipletMap::ClusterView::genLayerMap(const Layer& layer, const PartSch& part, const FetchSch& fetch, len_t batch_size, bool wgtB) const {
	if (mappers.size() == 1) {
		return const_cast<CoreMapper*>(mappers[0].first)->genLayerMap(layer, part, fetch, batch_size, wgtB);
	}
	// Every core type maps the same tile; the layer takes the slowest core's time and the sum of the cores' energies
	// (kept per core here: the engines multiply by the core count).
	CoreMapper::CoreMapping acc;
	acc.cost = CoreMapper::MapCost(0, 0);
	acc.ubuf = acc.buffer = acc.noc = acc.mac = 0;
	acc.util = acc.tot_util = 0;
	for (const auto& [mapper, n] : mappers) {
		CoreMapper::CoreMapping m = const_cast<CoreMapper*>(mapper)->genLayerMap(layer, part, fetch, batch_size, wgtB);
		if (!m.cost.is_valid()) return m;
		acc.cost.energy += m.cost.energy * n;
		acc.cost.time = std::max(acc.cost.time, m.cost.time);
		acc.ubuf += m.ubuf * n;
		acc.buffer += m.buffer * n;
		acc.noc += m.noc * n;
		acc.mac += m.mac * n;
		acc.util += m.util * n;
		acc.tot_util += m.tot_util * n;
	}
	acc.cost.energy /= total;
	acc.ubuf /= total;
	acc.buffer /= total;
	acc.noc /= total;
	acc.mac /= total;
	acc.util /= total;
	acc.tot_util /= total;
	return acc;
}

vol_t ChipletMap::min_ubuf(const Cluster& c) const {
	if (single()) return types[0].ul3;
	vol_t v = 0;
	bool first = true;
	for (const auto& [idx, n] : mix(c).counts) {
		(void)n;
		if (first || types[idx].ul3 < v) v = types[idx].ul3;
		first = false;
	}
	return v;
}

vol_t ChipletMap::min_ubuf() const {
	vol_t v = types[0].ul3;
	for (const ChipletType& t : types) v = std::min(v, t.ul3);
	return v;
}

vol_t ChipletMap::min_core_ubuf() const {
	vol_t v = types[0].core->ubuf().Size;
	for (const ChipletType& t : types) v = std::min(v, t.core->ubuf().Size);
	return v;
}

double ChipletMap::frequency_ghz() const {
	double f = types[0].tech.frequency_ghz;
	for (const ChipletType& t : types) f = std::min(f, t.tech.frequency_ghz);
	return f;
}

energy_t ChipletMap::noc_hop_cost() const {
	if (single()) return types[0].tech.noc_hop_cost;
	double sum = 0;
	double cores = 0;
	for (const ChipletType& t : types) {
		double n = static_cast<double>(t.num_chiplets) * cores_per_chiplet();
		sum += t.tech.noc_hop_cost * n;
		cores += n;
	}
	return sum / cores;
}

double ChipletMap::tops() const {
	if (single()) {
		const ChipletType& t = types[0];
		double tops = 16*t.vector_len*t.lane_len * Cluster::xlen * Cluster::ylen * 2;
		tops /= 1024;
		return tops;
	}
	long long macs = 0;
	for (const ChipletType& t : types) {
		macs += 16LL * t.vector_len * t.lane_len * t.num_chiplets * cores_per_chiplet() * 2;
	}
	return static_cast<double>(macs) / 1024;
}

void ChipletMap::compute_areas(const Technology& base, bw_t NoC_bw, bw_t NoP_bw) {
	density_t NoP_den = base.nop_den;
	double compute_die_num = NoC::x_cut * NoC::y_cut;
	for (ChipletType& t : types) {
		const Technology& T = t.tech;
		density_t SRAM_den = T.sram_den;
		density_t MAC_den = T.mac_den;//8-bit
		density_t LR_mac_den = MAC_den * 8;
		density_t NoC_den = T.noc_den;
		double post_layout_scale = T.post_layout_scale;
		double control_unit_prop = T.control_unit_prop;
		double DFT_prop = T.dft_prop;
		// upstream sizes the SRAM area with the Polar buffers of the MAC geometry, whichever core runs
		vol_t al1 = 8 * t.vector_len / 8 KB;
		vol_t ol1 = 2 * t.lane_len / 8 KB;
		vol_t wl1 = 4 * t.lane_len*t.vector_len / 64 KB;
		double sram_area_per_core = (t.ul3 + (al1 + wl1 + ol1)*16) * SRAM_den;
		double mac_area_per_core = 16*t.vector_len * t.lane_len * MAC_den + t.LR_mac_num*LR_mac_den;
		double NoC_area_per_core = NoC_bw * NoC_den;
		double core_area = (sram_area_per_core + mac_area_per_core + NoC_area_per_core)*post_layout_scale;
		double core_len = sqrt(core_area);
		double core_len_real = core_len;
		double D2D_area_per_compute_die=0;
		if (compute_die_num == 1) {
			D2D_area_per_compute_die = 0;
		}
		else if (compute_die_num == 2) {
			D2D_area_per_compute_die = 2 * NoC::y_step * NoP_den * NoP_bw;
		}
		else {
			D2D_area_per_compute_die = 2 * NoC::x_step * NoP_den * NoP_bw + 2 * NoC::y_step * NoP_den * NoP_bw;
		}
		double compute_die_area = core_len_real * NoC::x_step * core_len_real * NoC::y_step + D2D_area_per_compute_die;
		compute_die_area *= (control_unit_prop * DFT_prop);
		compute_die_area /= 1000000;
		t.core_area_um2 = core_area;
		t.compute_die_area_mm2 = compute_die_area;
	}
}

double ChipletMap::compute_die_area_mm2() const {
	if (single()) return types[0].compute_die_area_mm2;
	return total_compute_die_area_mm2() / num_chiplets();
}

double ChipletMap::total_compute_die_area_mm2() const {
	if (single()) return types[0].compute_die_area_mm2 * static_cast<double>(num_chiplets());
	double a = 0;
	for (const ChipletType& t : types) a += t.compute_die_area_mm2 * t.num_chiplets;
	return a;
}

double ChipletMap::max_compute_die_area_mm2() const {
	double a = 0;
	for (const ChipletType& t : types) a = std::max(a, t.compute_die_area_mm2);
	return a;
}

void ChipletMap::dump(std::ostream& os) const {
	os << "CHIPLETS: types=" << types.size() << " grid=" << (int)x_cut << "x" << (int)y_cut << " clock_ghz=" << frequency_ghz() << " map=";
	for (size_t i = 0; i < grid.size(); ++i) os << (i ? "," : "") << grid[i];
	os << std::endl;
	for (const ChipletType& t : types) {
		os << "CHIPLET_TYPE " << t.idx << ": node=" << t.tech.node << " frequency_ghz=" << t.tech.frequency_ghz << " mm=" << t.mm << " mac=" << t.mac_dim << " ul3_kb=" << (t.ul3 / 1024)
		   << " count=" << t.num_chiplets << " cores=" << t.num_chiplets * cores_per_chiplet() << " core_area_um2=" << t.core_area_um2
		   << " compute_die_area_mm2=" << t.compute_die_area_mm2 << std::endl;
	}
}
