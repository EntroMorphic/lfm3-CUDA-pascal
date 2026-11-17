#pragma once

namespace lfm {
// Call only after successful acquisition. A containing object's destructor does
// not run if its constructor throws, so release the acquired handle explicitly.
// Cleanup must be noexcept, just like an ordinary resource-owning destructor.
template<class Handle,class Cleanup,class Setup>
void finish_handle_setup(Handle& handle,Cleanup cleanup,Setup setup) {
  try {setup();}
  catch(...) {cleanup(handle);handle=Handle{};throw;}
}
} // namespace lfm
