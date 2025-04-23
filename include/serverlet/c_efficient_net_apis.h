//
// Created by user on 4/23/25.
//

#ifndef INFER_C_EFFICIENT_NET_APIS_H
#define INFER_C_EFFICIENT_NET_APIS_H

typedef unsigned char byte;

#ifdef __cplusplus
extern "C" {
#endif

void c_efficient_net_init(const char* engine_file_path, int maximum_batch_size);

void c_efficient_net_release();

bool c_efficient_net_add_image(int n_index, byte* cstr, int n_channels, int n_width, int n_height);

bool c_efficient_net_inference();

float* c_efficient_net_get_result(int n_index, int* n_size);

#ifdef __cplusplus
};
#endif

#endif //INFER_C_EFFICIENT_NET_APIS_H
