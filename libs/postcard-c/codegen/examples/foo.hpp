#ifndef Foo_H
#define Foo_H

#include <stdio.h>
#include <span>
#include <unordered_map>
#include <variant>
#include <vector>

#if __has_include("postcard.h")
#include "postcard.h"
#endif


struct Foo {
  uint32_t id;
  std::string name;
  std::vector<int64_t> values;
  bool is_active;
  std::vector<uint8_t> byte_arr;
  std::unordered_map<std::string, int32_t> metadata;
  


  size_t encoded_size() const {
    size_t size = 0;
    size += postcard_size_u32(id);
    size += postcard_size_string(name.length());
    size += postcard_size_seq(values.size());
    for(const auto& val: values) {
      size += postcard_size_i64(val);
    }
    size += postcard_size_bool();
    size += postcard_size_byte_array(byte_arr.size());
    size += postcard_size_map(metadata.size());
    for(const auto& [k, v]: metadata) {
      size += postcard_size_string(k.length());
      size += postcard_size_i32(v);
    }
    
    return size;
  }

  postcard_error_t encode(std::span<std::byte>& output) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(output.data()), output.size());
    auto res = encode_raw(&slice);
    if(res != POSTCARD_SUCCESS) return res;
    output = output.subspan(0, slice.len);
    return POSTCARD_SUCCESS;
  }

  std::vector<std::byte> encode_vec() {
    // Pre-allocate vector with the required size
    std::vector<std::byte> vec(encoded_size());
    
    // Create a span from the vector
    auto span = std::span<std::byte>(vec);
    
    // Encode into the span
    postcard_slice_t slice;
    postcard_init_slice(&slice, reinterpret_cast<uint8_t*>(span.data()), span.size());
    auto res = encode_raw(&slice);
    
    // Resize to actual used length if successful
    if (res == POSTCARD_SUCCESS) {
      vec.resize(slice.len);
    } else {
      vec.clear(); // Clear the vector on error
    }
    
    return vec;
  }

  postcard_error_t encode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_encode_u32(slice, id);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_string(slice, name.c_str(), name.length());
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_start_seq(slice, values.size());
    for(const auto& val: values) {
      result = postcard_encode_i64(slice, val);
    }
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_bool(slice, is_active);
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_encode_byte_array(slice, byte_arr.data(), byte_arr.size());
        if(result != POSTCARD_SUCCESS) return result;
    result = postcard_start_map(slice, metadata.size());
    for(const auto& [k, v]: metadata) {
      result = postcard_encode_string(slice, k.c_str(), k.length());
      result = postcard_encode_i32(slice, v);
      if(result != POSTCARD_SUCCESS) return result;
    }
        if(result != POSTCARD_SUCCESS) return result;
    
    return POSTCARD_SUCCESS;
  }

  postcard_error_t decode(std::span<const std::byte>& input) {
    postcard_slice_t slice;
    postcard_init_slice(&slice, const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(input.data())), input.size());
    postcard_error_t result = decode_raw(&slice);
    if (result == POSTCARD_SUCCESS) {
      // Update the input span to point past the decoded data
      input = input.subspan(slice.len);
    }
    return result;
  }

  postcard_error_t decode_raw(postcard_slice_t* slice) {
    postcard_error_t result;
    result = postcard_decode_u32(slice, &id);
    if(result != POSTCARD_SUCCESS) return result;

    size_t name_len;
    result = postcard_decode_string_len(slice, &name_len);
    if (result != POSTCARD_SUCCESS) return result;
    name.resize(name_len);
    if (name_len > 0) {
        result = postcard_decode_string(slice, name.data(), name_len, name_len);
        if (result != POSTCARD_SUCCESS) return result;
    }
    if(result != POSTCARD_SUCCESS) return result;

    size_t values_len;
    result = postcard_decode_seq_len(slice, &values_len);
    if (result != POSTCARD_SUCCESS) return result;
    values.clear();
    values.reserve(values_len);
    for(size_t i = 0; i < values_len; i++) {
        int64_t val;
        result = postcard_decode_i64(slice, &val);
        values.push_back(val);
    }
    if(result != POSTCARD_SUCCESS) return result;

    result = postcard_decode_bool(slice, &is_active);
    if(result != POSTCARD_SUCCESS) return result;

    size_t byte_arr_len;
    result = postcard_decode_byte_array_len(slice, &byte_arr_len);
    if (result != POSTCARD_SUCCESS) return result;
    byte_arr.resize(byte_arr_len);
    if (byte_arr_len > 0) {
        result = postcard_decode_byte_array(slice, byte_arr.data(), byte_arr_len, byte_arr_len);
        if (result != POSTCARD_SUCCESS) return result;
    }
    if(result != POSTCARD_SUCCESS) return result;

    size_t metadata_len;
    result = postcard_decode_map_len(slice, &metadata_len);
    if (result != POSTCARD_SUCCESS) return result;
    metadata.clear();
    for(size_t i = 0; i < metadata_len; i++) {
        std::string k;
        size_t k_len;
        result = postcard_decode_string_len(slice, &k_len);
        if (result != POSTCARD_SUCCESS) return result;
        k.resize(k_len);
        if (k_len > 0) {
            result = postcard_decode_string(slice, k.data(), k_len, k_len);
            if (result != POSTCARD_SUCCESS) return result;
        }
        if (result != POSTCARD_SUCCESS) return result;
        int32_t v;
        result = postcard_decode_i32(slice, &v);
        if (result != POSTCARD_SUCCESS) return result;
        metadata[k] = v;
    }
    if(result != POSTCARD_SUCCESS) return result;

    
    return POSTCARD_SUCCESS;
  }
};

#endif
