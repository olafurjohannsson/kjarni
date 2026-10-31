// kjarni.h before kjarni.hpp has to compile: Qt examples and C callers adding the
// wrapper include them in this order.
#include "kjarni.h"
#include "kjarni.hpp"

std::string_view include_order_version() { return kjarni::version(); }
