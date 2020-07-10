func @xor_training() {
  %c0 = hex.new.chain

  %lr = "dht.create_uninitialized_tensor.f32.1"() { shape = [1: i64] } : () -> !t.tensor
  %c1 = dht.fill_tensor_with_constant.f32 %lr, %c0 0.2 : f32

  %inputs = "dht.create_uninitialized_tensor.f32.2"() { shape = [4 : i64, 2 : i64] } : () -> !t.tensor
  %c2 = "dht.set_tensor_with_constant_values.f32"(%inputs, %c0)
    { values = [0.0 : f32, 0.0 : f32, 0.0 : f32, 1.0 : f32, 1.0 : f32, 0.0 : f32, 1.0 : f32, 1.0 : f32 ] } : (!t.tensor, !hex.chain) -> !hex.chain

  %expected_output = "dht.create_uninitialized_tensor.f32.2"() { shape = [4 : i64, 1 : i64] } : () -> !t.tensor
  %c3 = "dht.set_tensor_with_constant_values.f32"(%expected_output, %c0)
    { values = [0.0 : f32, 1.0 : f32, 1.0 : f32, 0.0 : f32 ] } : (!t.tensor, !hex.chain) -> !hex.chain

  %c4 = hex.merge.chains %c1, %c2, %c3

  %hidden_weights = "dht.create_uninitialized_tensor.f32.2"() { shape = [2 : i64, 2 : i64] } : () -> !t.tensor
  %hidden_bias    = "dht.create_uninitialized_tensor.f32.2"() { shape = [1 : i64, 2 : i64] } : () -> !t.tensor
  %output_weights = "dht.create_uninitialized_tensor.f32.2"() { shape = [2 : i64, 1 : i64] } : () -> !t.tensor
  %output_bias    = "dht.create_uninitialized_tensor.f32.2"() { shape = [1 : i64, 1 : i64] } : () -> !t.tensor

  // Other constants
  %one = hex.constant.f32 1.0
  %zero = hex.constant.f32 0.0

  tfrt_test.benchmark "mnist_training_benchmark"(
    %c0 : !hex.chain,
    %c4 : !hex.chain,

    %inputs : !t.tensor,
    %expected_output : !t.tensor,

    // Trainable variables.
    %hidden_weights : !t.tensor,
    %hidden_bias    : !t.tensor,
    %output_weights : !t.tensor,
    %output_bias    : !t.tensor,

    %lr : !t.tensor,

    %one : f32,
    %zero : f32
  )
  duration_secs = 100, max_count = 10000, num_warmup_runs = 0
  {
    ////////////////////////////////////////
    // Forward Pass.
    ////////////////////////////////////////
    %hidden_shape = ts.build_shape [4 : i64, 2 : i64]
    // hidden_layer_activation = hidden_bias
    %hidden_layer_activation = "tfrt_test.broadcast.f32.2"(%hidden_bias, %hidden_shape, %c4) : (!t.tensor, !ts.shape, !hex.chain) -> !t.tensor
    // hidden_layer_activation += 1 * inputs * hidden_weights
    %c5 = "eigen.matmul.f32"(%one, %inputs, %hidden_weights, %one, %hidden_layer_activation, %c4) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain
    // hidden_layer_activation = sigmoid(hidden_layer_activation)
    %c6 = "sigmoid_inplace.f32"(%hidden_layer_activation, %c5) : (!t.tensor, !hex.chain) -> !hex.chain

    %output_shape = ts.build_shape [4 : i64, 1 : i64]
    // predicted_output = output_bias
    %predicted_output = "tfrt_test.broadcast.f32.2"(%output_bias, %output_shape, %c6) : (!t.tensor, !ts.shape, !hex.chain) -> !t.tensor
    // predicted_output += 1 * hidden_layer_activation * output_weights
    %c7 = "eigen.matmul.f32"(%one, %hidden_layer_activation, %output_weights, %one, %predicted_output, %c6) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain
    // predicted_output = sigmoid(predicted_output)
    %c8 = "sigmoid_inplace.f32"(%predicted_output, %c7) : (!t.tensor, !hex.chain) -> !hex.chain

    ////////////////////////////////////////
    // Backward Pass.
    ////////////////////////////////////////
    %c9 = "tfrt_test.flatten.f32"(%predicted_output, %gradient, %c6) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c10 = "tfrt_test.subtract_inplace.f32"(%train_label_onehot, %gradient, %c9) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    // Get the mean for minibatch SGD
    %c11 = "tfrt_test.mean_axis_zero.f32"(%gradient, %gradient_b1, %c10) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c12 = hex.merge.chains %c1, %c11
    // Update b_1.
    %c13 = "tfrt_test.gradient_descent.f32"(%gradient_b1, %lr, %b_1, %c12) : (!t.tensor, !t.tensor, !t.tensor, !hex.chain) -> !hex.chain

    // Update W_1. Gradient descent implemented with GEMM: C = alpha * A * B + beta * C
    // Reference: http://cs231n.stanford.edu/handouts/linear-backprop.pdf
    %c14 = "tfrt_test.tensor_transpose.f32"(%activation_0, %transposed_activation_0, %c2) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c15 = hex.merge.chains %c10, %c14

    // Calculate transpose of W_1 before W_1 gets updated.
    %c16 = "tfrt_test.tensor_transpose.f32"(%w_1, %transposed_w_1, %c0) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c17 = hex.merge.chains %c10, %c16

    // Update W_1.
    %c18 = "eigen.matmul.f32"(%minus_lr_constant, %transposed_activation_0, %gradient, %one, %w_1, %c15) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain

    // Calculate gradient(a0).
    %c19 = "eigen.matmul.f32"(%one, %gradient, %transposed_w_1, %zero, %gradient_a0, %c17) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain
    %c20 = hex.merge.chains %c2, %c19
    %c21 = "tfrt_test.relu_grad_inplace.f32"(%activation_0, %gradient_a0, %c20) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain

    // Update b_0.
    %c22 = "tfrt_test.mean_axis_zero.f32"(%gradient_a0, %gradient_b0, %c21) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c23 = hex.merge.chains %c1, %c22
    %c24 = "tfrt_test.gradient_descent.f32"(%gradient_b0, %lr, %b_0, %c23) : (!t.tensor, !t.tensor, !t.tensor, !hex.chain) -> !hex.chain

    // Update W_0.
    %c25 = "tfrt_test.tensor_transpose.f32"(%train_image, %transposed_input_image, %c0) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain
    %c26 = hex.merge.chains %c21, %c25
    %c27 = "eigen.matmul.f32"(%minus_lr_constant, %transposed_input_image, %gradient_a0, %one, %w_0, %c26) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain

    hex.return %c27 : !hex.chain
  }

  hex.return
}
