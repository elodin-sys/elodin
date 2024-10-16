memserve is a web server that serves static web content from memory. It is somewhat unique in that it loads all content into memory at startup, and then serves it from there. It also does some pre-processing, such as gzip compression and ETag generation, to reduce the amount of work that needs to be done when handling requests. Obviously, this approach is not suitable for very large sites.

As a further optimization, memserve will prepare all possible responses ahead of time. As a result, it won't need to do any dynamic memory allocation when handling requests.

Due to its very limited scope, memserve will make several simplifying assumptions, which are also limitations:
- Only GET requests are supported.
- Only HTTP/1.1 is supported.
- Only strong ETags are supported, "Last-Modified" headers are not used.
- Only gzip compression is supported, and only for text-based content types.
- Accept headers are ignored.
- "If-None-Match" is the only conditional request header that is supported. "If-Modified-Since" is not supported.
- Chunked transfer encoding is not supported.
- "Connection" headers are ignored. Connections are always persistent and are closed after an idle timeout of 60s.
- No TLS support.
- Max URL length is 1KiB.
- Max request length is 8KiB.
- Max file path depth from content root is 8.
- No pipelining support.
- Single-threaded.
- Maximum file size is 32MiB.
- Maximum number of connections is 1024.
- "Date" header is not set in the response. It's technically required by the HTTP/1.1 spec, but I can argue section 14.8.3 applies to memserve.
