{
  stdenv,
  lib,
  cmake,
  pkg-config,
  protobuf,
  grpc,
  openssl,
  zlib,
  ...
}:
stdenv.mkDerivation {
  pname = "elodin-db-protos";
  version = "0.1.0";

  src = ../../libs/db/proto;

  nativeBuildInputs = [
    cmake
    pkg-config
    protobuf
    grpc
  ];

  propagatedBuildInputs = [
    protobuf
    grpc
    openssl
    zlib
  ];

  cmakeFlags = ["-DCMAKE_POSITION_INDEPENDENT_CODE=ON"];

  meta = {
    description = "C++ protobuf and gRPC library for Elodin DB";
    license = with lib.licenses; [mit asl20];
    platforms = lib.platforms.unix;
  };
}
