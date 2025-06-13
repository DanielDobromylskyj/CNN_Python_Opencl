
inline float relu(float x) {
    return fmax(0.0f, x);
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

__kernel void forward(__global float* inputs,
                      __global float* outputs,
                      __global float* weights,
                      int input_width,
                      int input_height,
                      int kernel_width,
                      int kernel_height,
                      int output_width,
                      int stride,
                      int channels
) {
    int output_x = get_global_id(0);
    int output_y = get_global_id(1);

    int input_x_anchor = output_x * stride;
    int input_y_anchor = output_y * stride;

    float total_sum = 0.0f;
    for (int channel=0; channel<channels; channel++) {
        int base_weight_index = kernel_width * kernel_height * channel;
        int base_input_index = input_width * input_height * channel;

        for (int dx=0; dx<kernel_width; dx++) {
            for (int dy=0; dy<kernel_height; dy++) {
                int weight_index = base_weight_index + dy * kernel_width) + dx;
                int input_index = base_input_index + ((input_y_anchor + dy) * input_height) + (input_x_anchor + dx);

                float weighted_value = weights[weight_index] * inputs[input_index];
                total_sum = total_sum + weighted_value;
            }
        }
    }

    int output_index = output_y * output_width + output_x;
    outputs[output_index] = total_sum;
}
