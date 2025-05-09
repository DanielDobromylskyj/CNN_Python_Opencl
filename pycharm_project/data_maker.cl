int index(int x, int y, int width) {
    return (y * width + x) * 3;
}


__kernel void rotate_180(__global float* buffer, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int offset_index = index(width - x - 1, height - y - 1, width);
    int true_index = index(x, y, width);

    float a = buffer[offset_index];
    float b = buffer[offset_index+1];
    float c = buffer[offset_index+2];

    buffer[offset_index] = buffer[true_index];
    buffer[offset_index+1] = buffer[true_index+1];
    buffer[offset_index+2] = buffer[true_index+2];

    buffer[true_index] = a;
    buffer[true_index+1] = b;
    buffer[true_index+2] = c;
}


__kernel void blur(__global float* buffer, int width, int height) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int true_index = index(x, y, width);

    for (int dx=-1; dx < 2; dx++) {
        for (int dy=-1; dy < 2; dy++) {
            if (dx == 0 && dy == 0) { continue; }

            if (x + dx > 0 && x + dx < width && y + dy > 0 && y + dy < height) {
                int offset_index = index(x + dx, y + dy, width);

                buffer[true_index] = buffer[true_index] + buffer[offset_index] * 0.1 * dx;
                buffer[true_index+1] = buffer[true_index+1] + buffer[offset_index+1] * 0.1 * dx;
                buffer[true_index+2] = buffer[true_index+2] + buffer[offset_index+2] * 0.1 * dx;
            }
        }
    }
}

__kernel void colour_shift(__global float* buffer, int width, int height, float shift) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    int i = index(x, y, width);

    buffer[i] = buffer[i] + shift;
    buffer[i+1] = buffer[i+1] + shift;
    buffer[i+2] = buffer[i+2] + shift;
}
