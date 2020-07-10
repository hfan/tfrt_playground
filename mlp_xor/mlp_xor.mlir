func @test_sigmoid() {
  %ch0 = hex.new.chain

  %activation = "dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64] } : () -> !t.tensor
  %ch1 = "dht.set_tensor_with_constant_values.f32"(%activation, %ch0)
    { values = [-1.0 : f32, 2.0 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %ch2 = "sigmoid_inplace.f32"(%activation, %ch1) : (!t.tensor, !hex.chain) -> !hex.chain

  %ch3 = dht.print_tensor %activation, %ch2

  hex.return
}

func @test_sigmoid_gradient() {
  %ch0 = hex.new.chain

  %activation = "dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64] } : () -> !t.tensor
  %ch1 = "dht.set_tensor_with_constant_values.f32"(%activation, %ch0)
    { values = [-1.0 : f32, 2.0 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %gradient = "dht.create_uninitialized_tensor.f32.2"()
    { shape = [2 : i64] } : () -> !t.tensor

  %ch2 = "sigmoid_inplace.f32"(%activation, %ch1) : (!t.tensor, !hex.chain) -> !hex.chain

  %ch3 = "sigmoid_grad_inplace.f32"(%activation, %gradient, %ch1) : (!t.tensor, !t.tensor, !hex.chain) -> !hex.chain

  %ch4 = dht.print_tensor %gradient, %ch3

  hex.return
}

func @xor_inference() {
  %c0 = hex.new.chain

  %lr = "dht.create_uninitialized_tensor.f32.1"() { shape = [1: i64] } : () -> !t.tensor
  %c1 = dht.fill_tensor_with_constant.f32 %lr, %c0 0.2 : f32

  %inputs = "dht.create_uninitialized_tensor.f32.2"() { shape = [4 : i64, 2 : i64] } : () -> !t.tensor
  %c2 = "dht.set_tensor_with_constant_values.f32"(%inputs, %c0)
    { values = [0.0 : f32, 0.0 : f32, 0.0 : f32, 1.0 : f32, 1.0 : f32, 0.0 : f32, 1.0 : f32, 1.0 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %expected_output = "dht.create_uninitialized_tensor.f32.2"() { shape = [4 : i64, 1 : i64] } : () -> !t.tensor
  %c3 = "dht.set_tensor_with_constant_values.f32"(%expected_output, %c0)
    { values = [0.0 : f32, 1.0 : f32, 1.0 : f32, 0.0 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %hidden_weights = "dht.create_uninitialized_tensor.f32.2"() { shape = [2 : i64, 2 : i64] } : () -> !t.tensor
  %c4 = "dht.set_tensor_with_constant_values.f32"(%hidden_weights, %c0)
    { values = [4.11206598 : f32, 6.20361774 : f32, 4.10740065 : f32, 6.18121127 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %hidden_bias    = "dht.create_uninitialized_tensor.f32.2"() { shape = [2 : i64] } : () -> !t.tensor
  %c5 = "dht.set_tensor_with_constant_values.f32"(%hidden_bias, %c0)
    { values = [-6.30204227 : f32, -2.67542759 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %output_weights = "dht.create_uninitialized_tensor.f32.2"() { shape = [2 : i64, 1 : i64] } : () -> !t.tensor
  %c6 = "dht.set_tensor_with_constant_values.f32"(%output_weights, %c0)
    { values = [-9.17221319 : f32, 8.46304139 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %output_bias    = "dht.create_uninitialized_tensor.f32.2"() { shape = [1 : i64, 1 : i64] } : () -> !t.tensor
  %c7 = "dht.set_tensor_with_constant_values.f32"(%output_bias, %c0)
    { values = [-3.85270514 : f32] } : (!t.tensor, !hex.chain) -> !hex.chain

  %one = hex.constant.f32 1.0

  %c8 = hex.merge.chains %c1, %c2, %c3, %c4, %c5, %c6, %c7

  ////////////////////////////////////////
  // Forward Pass.
  ////////////////////////////////////////
  %hidden_shape = ts.build_shape [4 : i64, 2 : i64]
  // hidden_layer_activation = hidden_bias
  %hidden_layer_activation = "tfrt_test.broadcast.f32.2"(%hidden_bias, %hidden_shape, %c8) : (!t.tensor, !ts.shape, !hex.chain) -> !t.tensor
  // hidden_layer_activation += 1 * inputs * hidden_weights
  %c9 = "eigen.matmul.f32"(%one, %inputs, %hidden_weights, %one, %hidden_layer_activation, %c8) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain
  // hidden_layer_activation = sigmoid(hidden_layer_activation)
  %c10 = "sigmoid_inplace.f32"(%hidden_layer_activation, %c9) : (!t.tensor, !hex.chain) -> !hex.chain

  %output_shape = ts.build_shape [4 : i64, 1 : i64]
  // predicted_output = output_bias
  %predicted_output = "tfrt_test.broadcast.f32.2"(%output_bias, %output_shape, %c10) : (!t.tensor, !ts.shape, !hex.chain) -> !t.tensor
  // predicted_output += 1 * hidden_layer_activation * output_weights
  %c11 = "eigen.matmul.f32"(%one, %hidden_layer_activation, %output_weights, %one, %predicted_output, %c10) : (f32, !t.tensor, !t.tensor, f32, !t.tensor, !hex.chain) -> !hex.chain
  // predicted_output = sigmoid(predicted_output)
  %c12 = "sigmoid_inplace.f32"(%predicted_output, %c11) : (!t.tensor, !hex.chain) -> !hex.chain

  %c13 = dht.print_tensor %predicted_output, %c12

  hex.return
}
