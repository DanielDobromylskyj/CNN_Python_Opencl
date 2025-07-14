
inline float relu(float x) {
    return fmax(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__kernel void forward(__global float* inputs, __global float* outputs,
                      __global float* weights, int input_size, int output_size) {

    int node_in_index = get_global_id(0);
    int node_out_index = get_global_id(1);
    int batch_index = get_global_id(2);

    int batch_input_offset = input_size * batch_index;
    int batch_output_offset = batch_index * input_size * output_size;

    int weight_index = node_out_index * input_size + node_in_index;

    float unactivated = inputs[node_in_index + batch_input_offset] * weights[weight_index];

    // As we are going to reduce the values later, we will store every value separately,
    // We use the weight index to index the output array, so we get no overlapping values
    outputs[weight_index + batch_output_offset] = unactivated;
}


__kernel void reduce_outputs(__global float* unreduced_outputs,
                             __global float* reduced_outputs,
                             __global float* biases,
                             int input_size, int output_size,
                             int activation_type) {

    int output_index = get_global_id(0);
    int batch_index = get_global_id(1);

    int batch_input_offset = batch_index * input_size * output_size;
    int batch_output_offset = batch_index * output_size;

    float local_sum = 0.0; // the indexing here I think is wrong
    for (int input_index = 0; input_index < input_size; input_index++) {
        int array_index = output_index * input_size + input_index + batch_input_offset;
        local_sum += unreduced_outputs[array_index];
    }

    local_sum += biases[output_index];
    //unactivated_outputs[output_index] = local_sum;  // Only used for training version. This version is optimised for speed.

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

    reduced_outputs[output_index + batch_output_offset] = activated;
}
