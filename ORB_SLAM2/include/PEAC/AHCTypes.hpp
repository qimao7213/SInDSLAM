// Copyright (C) 2014,2023 Mitsubishi Electric Research Laboratories (MERL)
//
// SPDX-License-Identifier: AGPL-3.0-or-later
#pragma once

#ifndef USE_BOOST_SHARED_PTR
#include <memory>
#else
#include <boost/shared_ptr.hpp>
#endif

namespace ahc {
#ifndef USE_BOOST_SHARED_PTR
	using std::shared_ptr;
#else
	using boost::shared_ptr;
#endif
}
