
inline float relu(float x) {
    return fmax(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__kernel void forward(__global float* inputs,
                      __global float* outputs,
                      __global float* weights,
                      __global float* biases,
                      int input_width,
                      int input_height,
                      int kernel_width,
                      int kernel_height,
                      int output_width,
                      int output_height,
                      int stride,
                      int channels,
                      int activation_type
) {
    int output_x = get_global_id(0);
    int output_y = get_global_id(1);
    int batch_index = get_global_id(2);

    int batch_output_offset = output_width * output_height * batch_index;
    int batch_input_offset = input_width * input_height * channels * batch_index;

    int output_index = output_y * output_width + output_x;
    int input_x_anchor = output_x * stride;
    int input_y_anchor = output_y * stride;

    // fixme - Batch indexing is fucked, At least the weight index is fine (I didn't touch that bit)

    float total_sum = biases[output_index];
    for (int channel=0; channel < channels; channel++) {
        int base_weight_index = kernel_width * kernel_height * channel;
        int base_input_index = (input_width * input_height * channel) + batch_input_offset;

        for (int dx=0; dx<kernel_width; dx++) {
            for (int dy=0; dy<kernel_height; dy++) {
                int weight_index = base_weight_index + (dy * kernel_width) + dx;
                int input_index = base_input_index + ((input_y_anchor + dy) * input_width) + (input_x_anchor + dx);

                float weighted_value = weights[weight_index] * inputs[input_index];
                total_sum = total_sum + weighted_value;
            }
        }
    }

    float activated = 0.0;
    switch (activation_type) {
            case 1:  // ReLU
                activated = relu(total_sum);
                break;
            case 2:  // Sigmoid
                activated = sigmoid(total_sum);
                break;
            default: // Default
                activated = total_sum;
                break;
    }


    outputs[output_index + batch_output_offset] = activated;
}
