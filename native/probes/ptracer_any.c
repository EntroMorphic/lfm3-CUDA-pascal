// validation/prefill-stall-v1: preloaded into a process so that a debugger may attach to it later (Thor's ptrace_scope
// 1 lets only an ancestor trace; the process itself may opt in with PR_SET_PTRACER). Diagnostics only: never part of
// a build of the engine. Build: gcc -shared -fPIC -O2 -o libptracer_any.so ptracer_any.c
#include <sys/prctl.h>
#ifndef PR_SET_PTRACER
#define PR_SET_PTRACER 0x59616d61
#endif
#ifndef PR_SET_PTRACER_ANY
#define PR_SET_PTRACER_ANY ((unsigned long)-1)
#endif
__attribute__((constructor)) static void allow_any_tracer(void) { prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0); }
