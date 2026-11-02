#ifndef SERVED_H
#define SERVED_H

// chess/main (L7, H-E5): shares priced by a model outside Gemini.
//
// With GEMINI_CORE_SERVER="<write fd> <read fd>" in the environment, main.cpp wraps every chiplet
// type's mapper in a ServedMapper. A share (one core's convolution workload, the argument of
// genMapping) the wrapper has not seen goes to the process at the other end of the two pipes as
// one line, and its mapping comes back as one line:
//
//   request:  share <type> <C> <K> <R> <S> <H> <W> <sH> <sW> <B> <nGroup>
//   answer:   <energy> <time> <ubuf> <buffer> <noc> <mac> <util> <tot_util>   the share's CoreMapping
//             -                                                             the type's own mapper prices it
//
// Answers are kept per type and share, so a share costs one round trip however often the search
// asks for it. Everything else stays the type's own mapper's: a layer's unit time (set_utime), the
// buffer size and lane-wise layers. Gemini's search is single-threaded, so one pair of pipes serves
// every type. At exit the server prints its counts (CORE_SERVER line) on stdout.

#include <array>
#include <cstdio>
#include <map>
#include <memory>

#include "coremapping.h"

class CoreServer {
public:
	// The server named by GEMINI_CORE_SERVER, or nullptr when the variable is unset.
	// Throws std::runtime_error when the variable is malformed or a descriptor cannot be opened.
	static std::unique_ptr<CoreServer> from_env();
	CoreServer(int write_fd, int read_fd);
	~CoreServer();
	// The share's mapping from the other end: true with `out` set, false when the type's own mapper is to price it.
	// Throws std::runtime_error when the other end closes or answers something else.
	bool ask(int type, const CoreMapper::ConvWl& wl, CoreMapper::CoreMapping& out);
	void count(bool answered) { ++asked; if (!answered) ++own; }
	void report() const;
private:
	FILE* to;
	FILE* from;
	unsigned long long asked = 0, distinct = 0, own = 0;
};

class ServedMapper : public CoreMapper {
public:
	ServedMapper(std::unique_ptr<CoreMapper> inner, int type, CoreServer& server);
	CoreMapping genMapping(const ConvWl& wl) override;
	void set_conv_utime(ConvLayer& l) const override;
	void set_lr_utime(LRLayer& l) const override;
	vol_t get_ubuf_size() const override;
private:
	typedef std::array<std::uint64_t, 10> Key;
	struct Answer {
		bool served;
		CoreMapping mapping;
	};
	std::unique_ptr<CoreMapper> inner;
	int type;
	CoreServer& server;
	std::map<Key, Answer> answers;
};

#endif // SERVED_H
