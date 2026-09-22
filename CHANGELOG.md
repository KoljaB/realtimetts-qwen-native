# CPU native 0.4.0

Adds optional streaming decoder overlap through a separate CPU backend and
worker pool. qt_init_cpu_ex accepts codec worker count, streaming cadence and
optional logical-CPU affinity masks while retaining ABI 5 and the legacy
initializer. Defaults remain serial and unpinned.

QWENTTS_CPU_STARTUP_PRIORITY=second_chunk temporarily parks generation for the
second two-frame decode block, then resumes overlap. It is off by default,
requires overlap and a two/four-frame cadence, and leaves model weights,
precision and sampling unchanged. Invalid settings fail before model loading.

Callbacks stay on the generation thread; pause acknowledgement parks native
workers, cancellation polling does not acknowledge pause, and context teardown
joins outstanding decode work.

Measured Linux CPU RTF decreased 23-26% in three paired playback cases. The
startup option recovered 18-23 ms of audible latency with a 2-6% generation-time
cost in the measured cases. Tested PCM was identical. Performance depends on
CPU topology and workload; the 80 ms deployment buffer is not a new default.

GPU startup experiments are excluded. Supported wheel targets remain Linux
x86-64, Windows x86-64, Intel Mac and Apple Silicon. Each new platform artifact
must pass the existing fresh-install speech gate before publication.
