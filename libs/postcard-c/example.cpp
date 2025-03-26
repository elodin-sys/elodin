#include "postcard.h"
#include <format>
#include <iostream>
#include <span>
#include <stdio.h>

int main() {
  // Create a buffer and a slice
  uint8_t buffer[128];
  postcard_slice_t slice;
  postcard_init_slice(&slice, buffer, sizeof(buffer));

  //
  // struct Foo {
  //     id: u32,
  //     name: String,
  //     values: Vec<i64> // len 3
  //     is_active: bool
  // }

  // encode id
  postcard_encode_u32(&slice, 1234);

  // encode name
  const char *name = "PostcardTest";
  postcard_encode_string(&slice, name, strlen(name));

  // encode the 3 values
  postcard_start_seq(&slice, 3);
  postcard_encode_i16(&slice, -10);
  postcard_encode_i16(&slice, 20);
  postcard_encode_i16(&slice, -30);

  // encode is_active
  postcard_encode_bool(&slice, true);

  // Print the encoded data
  std::span<uint8_t> data = std::span(slice.data, slice.len);
  std::println("serialized data {}", data);

  // Now let's decode this data
  postcard_slice_t decode_slice;
  postcard_init_slice(&decode_slice, buffer, slice.len);

  // Decode each field
  uint32_t id;
  postcard_decode_u32(&decode_slice, &id);
  std::println("id: {}", id);

  char name_buffer[64];
  size_t actual_len;
  postcard_decode_string_len(&decode_slice, &actual_len);
  postcard_decode_string(&decode_slice, name_buffer, sizeof(name_buffer),
                         actual_len);
  std::string decoded_name(name_buffer, actual_len);
  std::println("name: {}", decoded_name);

  // Decode the sequence
  size_t seq_len;
  postcard_decode_seq_len(&decode_slice, &seq_len);
  std::println("values len: {}", seq_len);
  std::vector<uint16_t> values;
  for (size_t i = 0; i < seq_len; i++) {
    int16_t value;
    values.push_back(postcard_decode_i16(&decode_slice, &value));
  }
  std::println("values: {}", values);

  // Decode the boolean bool is_active;
  bool is_active;
  postcard_decode_bool(&decode_slice, &is_active);
  std::println("is_active: {}", is_active);
  return 0;
}
