# tfrt_playground


This is a workspace for myself to explore the TFRT codebase, and experiment ideas to understand it’s design properties.

TFRT is the new runtime for TensorFlow, and it has been open-sourced recently.

If you are as excited as me after watching the TFRT introductory video from the TF Dev Summit of this year, or after reading the corresponding [blog post](https://blog.tensorflow.org/2020/04/tfrt-new-tensorflow-runtime.html), I highly recommend the [TFRT deep dive presentation](https://drive.google.com/drive/folders/1fkLJuVP-tIk4GENBu2AgemF3oXYGr2PB), which covers the design principle of the TFRT Host Runtime and a lot of the details too.

I have worked on the runtime of a SQL database, and the TFRT shares a lot of similarity in the regard of supporting a graph based execution plan, and also it is probably widely known that compilers and SQL optimizers have a lot in common. Many tricks are shared among runtime systems, such as operator fusion, just-in-time code generation, by-passing overhead for small queries/tasks. I’m eager to see their usages and effects in TFRT.

However, the design of TFRT has to meet the requirements that are not typically seen for a database server, for instance running on a wide variety of hardware, large number of types of operators/kernels, supporting both graph and eager execution.

The performance result of ResNet-50 demonstrated in TF Dev Summit shows the promising potential of the approach. I’m really excited about it that I made this repository for the following experiments:

1. A MLP network with inference and training (WIP) for a simple XOR computation. It involves adding a Sigmoid kernel into the `bef_executor`, and writing MLIR programs to construct the MLP network and perform inference/training.

    This is an effort to get familiarized with part of the TFRT workflow. Writing MLIR programs by hand is tedious, and it will be replaced by the compiler per TFRT roadmap, but it is a good exercise to see what the runtime expects at this level.

    I also noticed that the compilation time could be an issue for kernel development, as re-building `bef_executor` is somewhat time consuming. So I filed a [ticket](https://github.com/tensorflow/runtime/issues/13) for it.

2. An attempt to benchmark TFRT thread scheduling performance by solving the N-Queue problem using threads with TFRT.

    TFRT makes heavy use of AsyncValue backed by a thread pool to orchestrate the execution of the kernels, it would be interesting to know its overhead. So the focus is not to optimize N-Queue solving function with any algorithm or coding optimizations. The N-Queue problem is selected because its recursive nature makes it easy to generate many tasks to stress the thread scheduling.

    To provide the performance baseline, a standard C++ solution with an explicit task_queue and thread_pool and a single-threaded version are implemented.
