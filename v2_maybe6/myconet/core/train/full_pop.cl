inline float relu(float x) {
    return fmax(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

inline float derivative_sigmoid(float x) {
    return sigmoid(x) * (1 - sigmoid(x));
}

inline float derivative_relu(float x) {
    return (x > 0) ? 1.0f : 0.0f;
}

inline float clip(float value, float clip_value) {
    return fmin(fmax(value, -clip_value), clip_value);
}

__kernel void reduce_outputs_forward(__global float* unreduced_outputs,
                             __global float* reduced_outputs,
                             __global float* unactivated_outputs,
                             __global float* biases,
                             int input_size, int output_size,
                             int activation_type) {

    int output_index = get_global_id(0);

    float local_sum = 0.0; // the indexing here I think is wrong
    for (int input_index = 0; input_index < input_size; input_index++) {
        int array_index = output_index * input_size + input_index;
        local_sum += unreduced_outputs[array_index];
    }

    local_sum += biases[output_index];
    unactivated_outputs[output_index] = local_sum;

    float activated = 0.0;
    switch (activation_type) {
            case 1:  // ReLU
                activated = relu(local_sum);
                break;
            case 2:  // Sigmoid
                activated = sigmoid(local_sum);
                break;
            default: // Default
                activated = local_sum;
                break;
    }

    reduced_outputs[output_index] = activated;
}


// REMEMBER THAT WE ARE WORKING BACKWARDS. SO OUR "inputs" ARE THE RIGHT SIDE NODES OF A LAYER
__kernel void backwards(
        __global float* left_hand_nodes,
        __global float* right_hand_nodes_activated,
        __global float* right_hand_nodes_unactivated,
        __global float* weights,
        __global float* right_hand_nodes_error_gradients, // left_hand_nodes_error_gradients_unreduced
        __global float* left_hand_nodes_error_gradients_unreduced, // Same size as the weights, so we can "reduce" it later for our final output
        __global float* weight_gradients,
        __global float* bias_gradients,
        int right_hand_nodes_size, int left_hand_nodes_size, int activation_type,
        float learning_rate
) {
    int left_hand_index = get_global_id(0);
    int right_hand_index = get_global_id(1);

    float activated_value = right_hand_nodes_activated[right_hand_index];
    float unactivated_value = right_hand_nodes_activated[right_hand_index];

    float derivative = 1.0f;
    switch (activation_type) {
        case 1: // ReLU activation
            derivative = unactivated_value > 0 ? 1.0f : 0.0f;
            break;
        case 2: // Sigmoid activation
          derivative = activated_value * (1.0f - activated_value);
           break;
        default:
            derivative = 1.0f; // Linear activation (default)
            break;
    }

    float delta = right_hand_nodes_error_gradients[right_hand_index] * derivative;
    float weight_gradient = delta * left_hand_nodes[left_hand_index] * learning_rate;

    int weight_index = left_hand_index * right_hand_nodes_size + right_hand_index;

    weight_gradients[weight_index] = clip(weight_gradient, 1.0f);

    left_hand_nodes_error_gradients_unreduced[weight_index] = clip(weights[weight_index] * delta, 1.0f);

    if (left_hand_index == 0) {
        bias_gradients[right_hand_index] = delta * learning_rate;
    }
}

