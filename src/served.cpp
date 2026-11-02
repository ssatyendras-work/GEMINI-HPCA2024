#include "served.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

// chess/main (L7, H-E5): see served.h for the protocol.

std::unique_ptr<CoreServer> CoreServer::from_env() {
	const char* env = getenv("GEMINI_CORE_SERVER");
	if (!env || !*env) return nullptr;
	std::istringstream s(env);
	int w = -1, r = -1;
	if (!(s >> w >> r) || w < 0 || r < 0) throw std::runtime_error(std::string("GEMINI_CORE_SERVER must be \"<write fd> <read fd>\", not \"") + env + "\"");
	return std::make_unique<CoreServer>(w, r);
}

CoreServer::CoreServer(int write_fd, int read_fd) {
	to = fdopen(write_fd, "w");
	from = fdopen(read_fd, "r");
	if (!to || !from) throw std::runtime_error("GEMINI_CORE_SERVER: cannot open descriptors " + std::to_string(write_fd) + " and " + std::to_string(read_fd));
}

CoreServer::~CoreServer() {
	fclose(to);  // the other end sees the end of the requests
	fclose(from);
}

bool CoreServer::ask(int type, const CoreMapper::ConvWl& wl, CoreMapper::CoreMapping& out) {
	++distinct;
	fprintf(to, "share %d %u %u %u %u %u %u %u %u %u %u\n", type, wl.C, wl.K, wl.R, wl.S, wl.H, wl.W, wl.sH, wl.sW, wl.B, wl.nGroup);
	fflush(to);
	char line[512];
	if (!fgets(line, sizeof line, from)) throw std::runtime_error("GEMINI_CORE_SERVER: the server closed before answering a share of type " + std::to_string(type));
	if (line[0] == '-') return false;
	double v[8];
	char* p = line;
	for (double& x : v) {
		char* end;
		x = strtod(p, &end);  // 17 significant digits round-trip a double exactly
		if (end == p) throw std::runtime_error(std::string("GEMINI_CORE_SERVER: bad answer: ") + line);
		p = end;
	}
	out.cost = CoreMapper::MapCost(v[0], static_cast<cycle_t>(v[1]));
	out.ubuf = v[2];
	out.buffer = v[3];
	out.noc = v[4];
	out.mac = v[5];
	out.util = v[6];
	out.tot_util = v[7];
	return true;
}

void CoreServer::report() const {
	std::cout << "CORE_SERVER: " << asked << " asked, " << distinct << " distinct, " << own << " by the type's mapper" << std::endl;
}

ServedMapper::ServedMapper(std::unique_ptr<CoreMapper> _inner, int _type, CoreServer& _server)
	: CoreMapper(_inner->core()), inner(std::move(_inner)), type(_type), server(_server) {}

CoreMapper::CoreMapping ServedMapper::genMapping(const ConvWl& wl) {
	Key k{wl.C, wl.K, wl.R, wl.S, wl.H, wl.W, wl.sH, wl.sW, wl.B, wl.nGroup};
	auto it = answers.find(k);
	if (it == answers.end()) {
		Answer a;
		a.served = server.ask(type, wl, a.mapping);
		it = answers.emplace(k, a).first;
	}
	server.count(it->second.served);
	return it->second.served ? it->second.mapping : inner->genMapping(wl);
}

void ServedMapper::set_conv_utime(ConvLayer& l) const {
	inner->set_conv_utime(l);
}

void ServedMapper::set_lr_utime(LRLayer& l) const {
	inner->set_lr_utime(l);
}

vol_t ServedMapper::get_ubuf_size() const {
	return inner->get_ubuf_size();
}
