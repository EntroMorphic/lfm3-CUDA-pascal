// Image decoding for the vision front end: see lfm/image_decode.hpp for what it reproduces and why.
#include <lfm/image_decode.hpp>

#include <algorithm>
#include <array>
#include <csetjmp>
#include <cstdio>
#include <cstring>
#include <optional>
#include <string_view>

extern "C" {
#include <jpeglib.h>
}

#define STBI_ONLY_PNG
#define STBI_NO_STDIO
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#define STB_IMAGE_STATIC
#define STB_IMAGE_IMPLEMENTATION
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wsign-compare"
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif
#include "../third_party/stb/stb_image.h"
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

namespace lfm::image {
namespace {

using Bytes = std::vector<std::uint8_t>;

[[noreturn]] void fail(const char* code, const std::string& message) { throw DecodeError(code, message); }

bool starts_with(const std::uint8_t* p, std::size_t n, std::string_view prefix) {
  return n >= prefix.size() && std::memcmp(p, prefix.data(), prefix.size()) == 0;
}

void check_pixels(std::uint64_t w, std::uint64_t h, const Limits& limits) {
  if (w == 0 || h == 0) fail("corrupt_image", "image has a zero dimension");
  const std::uint64_t limit = std::min(limits.max_pixels, pillow_bomb_pixels);
  if (w * h > limit) {
    fail("image_too_large", "image has " + std::to_string(w * h) + " pixels, more than the limit of " +
                                std::to_string(limit));
  }
}

// ---------------------------------------------------------------- orientation (Pillow's getexif + exif_transpose)

// Tag 0x0112 of IFD0 in an EXIF block, read as Pillow 13.0.0 reads it (Image.Exif.load,
// TiffImagePlugin.ImageFileDirectory_v2.load and _setitem, ImageOps.exif_transpose), checked against Pillow on a grid of
// well-formed and malformed entries (tools/make_exif_orientation_grid.py). Pillow's rules are the TIFF specification
// read leniently where the intent is clear, so they are kept:
//   - up to two "Exif\0\0" prefixes are dropped; the header is II*\0, MM\0* or the swapped II\0* / MM*\0 (byte order
//     from the first two bytes); IFD0 at the header's offset. (A header that is none of these makes Pillow raise and HF
//     refuse the whole image; here the EXIF is ignored and the image shown: the pixels are not at fault. BigTIFF is
//     treated the same way.)
//   - entries in order; a tag already loaded is skipped; an unsupported type is skipped; data over 4 bytes lies at an
//     offset: if the running total of such data exceeds the block, the entry is skipped, and if it runs past the end
//     of the block, the reading stops there (entries already loaded stand); an entry with no data is skipped; an IFD
//     cut short stops the reading.
//   - the value: BYTE, UNDEFINED and ASCII give bytes or text, which match no orientation; the numeric types give
//     their first value, which counts when it equals an integer (6.0, or a rational 12/2, are 6).
// Returns the orientation (0 when the tag is present but matches none, which still blocks the XMP fallback), or
// nullopt when the tag is absent.
std::optional<int> tiff_orientation(const std::uint8_t* p, std::size_t n) {
  for (int k = 0; k < 2 && starts_with(p, n, std::string_view("Exif\0\0", 6)); ++k) {
    p += 6;
    n -= 6;
  }
  if (n < 8) return std::nullopt;
  const bool big = p[0] == 'M' && p[1] == 'M';
  const bool tiff = (p[0] == 'I' && p[1] == 'I' && ((p[2] == 42 && p[3] == 0) || (p[2] == 0 && p[3] == 42))) ||
                    (big && ((p[2] == 0 && p[3] == 42) || (p[2] == 42 && p[3] == 0)));
  if (!tiff) return std::nullopt;
  auto u16 = [&](std::size_t o) -> std::uint32_t {
    return big ? (std::uint32_t(p[o]) << 8 | p[o + 1]) : (std::uint32_t(p[o + 1]) << 8 | p[o]);
  };
  auto u32 = [&](std::size_t o) -> std::uint32_t {
    return big ? (std::uint32_t(p[o]) << 24 | std::uint32_t(p[o + 1]) << 16 | std::uint32_t(p[o + 2]) << 8 | p[o + 3])
               : (std::uint32_t(p[o + 3]) << 24 | std::uint32_t(p[o + 2]) << 16 | std::uint32_t(p[o + 1]) << 8 | p[o]);
  };
  auto unit = [](std::uint32_t type) -> int {  // Pillow's _load_dispatch sizes; 0: unsupported
    switch (type) {
      case 1: case 2: case 6: case 7: return 1;
      case 3: case 8: return 2;
      case 4: case 9: case 11: case 13: return 4;
      case 5: case 10: case 12: case 16: return 8;
      default: return 0;
    }
  };
  const std::size_t ifd = u32(4);
  if (ifd > n || n - ifd < 2) return std::nullopt;
  const std::uint32_t count = u16(ifd);
  std::vector<bool> loaded(65536);  // tags already loaded (Pillow's _tagdata); constant time per entry
  std::uint64_t total = 0;
  const std::uint8_t* data = nullptr;
  std::uint32_t data_type = 0;
  for (std::uint32_t i = 0; i < count; ++i) {
    const std::size_t e = ifd + 2 + std::size_t(i) * 12;
    if (e + 12 > n) break;  // _ensure_read fails: the reading stops
    const std::uint32_t tag = u16(e), type = u16(e + 2), values = u32(e + 4);
    if (loaded[tag]) continue;
    const int size1 = unit(type);
    if (size1 == 0) continue;
    const std::uint64_t size = std::uint64_t(values) * std::uint64_t(size1);
    const std::uint8_t* d = p + e + 8;
    if (size > 4) {
      const std::uint64_t off = u32(e + 8);
      total += size;
      if (total > n) continue;                // "Tag data larger than file": skipped
      if (off + size > n) break;              // _safe_read: "Truncated File Read", the reading stops
      d = p + off;
    }
    if (size == 0) continue;
    loaded[tag] = true;
    if (tag == 0x0112) {
      data = d;
      data_type = type;
    }
  }
  if (!data) return std::nullopt;
  auto integral = [](double v) -> int { return (v >= 1 && v <= 8 && v == double(int(v))) ? int(v) : 0; };
  switch (data_type) {
    case 1: case 2: case 7: return 0;  // bytes or text
    case 3: return integral(u16(0 + std::size_t(data - p)));
    case 4: case 13: return integral(u32(std::size_t(data - p)));
    case 6: return integral(std::int8_t(data[0]));
    case 8: return integral(std::int16_t(u16(std::size_t(data - p))));
    case 9: return integral(std::int32_t(u32(std::size_t(data - p))));
    case 11: {
      const std::uint32_t b = u32(std::size_t(data - p));
      float f;
      std::memcpy(&f, &b, 4);
      return integral(f);
    }
    case 12: case 16: {
      const std::size_t o = std::size_t(data - p);
      const std::uint64_t hi = u32(o), lo = u32(o + 4);
      const std::uint64_t b = big ? (hi << 32 | lo) : (lo << 32 | hi);
      if (data_type == 16) return b >= 1 && b <= 8 ? int(b) : 0;
      double v;
      std::memcpy(&v, &b, 8);
      return integral(v);
    }
    case 5: case 10: {  // IFDRational: num / den, NaN when den is 0
      const std::size_t o = std::size_t(data - p);
      const std::int64_t num = data_type == 5 ? std::int64_t(u32(o)) : std::int64_t(std::int32_t(u32(o)));
      const std::int64_t den = data_type == 5 ? std::int64_t(u32(o + 4)) : std::int64_t(std::int32_t(u32(o + 4)));
      if (den == 0 || num % den != 0) return 0;
      const std::int64_t v = num / den;
      return v >= 1 && v <= 8 ? int(v) : 0;
    }
  }
  return 0;
}

// Pillow's XMP fallback: the first match of tiff:Orientation(="|>)([0-9]).
std::optional<int> xmp_orientation(const std::uint8_t* p, std::size_t n) {
  static constexpr std::string_view key = "tiff:Orientation";
  for (std::size_t i = 0; i + key.size() + 2 < n + 1; ++i) {
    if (std::memcmp(p + i, key.data(), key.size()) != 0) continue;
    std::size_t j = i + key.size();
    if (j + 1 < n && p[j] == '=' && p[j + 1] == '"') j += 2;
    else if (j < n && p[j] == '>') j += 1;
    else continue;
    if (j < n && p[j] >= '0' && p[j] <= '9') return p[j] - '0';
  }
  return std::nullopt;
}

struct Metadata {
  Bytes exif;          // Pillow's info["exif"] (JPEG: the APP1 segments concatenated; PNG: "Exif\0\0" + eXIf)
  bool has_exif = false;
  Bytes raw_profile;   // PNG "Raw profile type exif", decoded from hex
  bool has_raw_profile = false;
  Bytes xmp;           // JPEG APP1 XMP or PNG iTXt XML:com.adobe.xmp
  bool has_xmp = false;
};

int orientation_of(const Metadata& m) {
  std::optional<int> o;
  if (m.has_exif) o = tiff_orientation(m.exif.data(), m.exif.size());
  else if (m.has_raw_profile) o = tiff_orientation(m.raw_profile.data(), m.raw_profile.size());
  if (!o && m.has_xmp) o = xmp_orientation(m.xmp.data(), m.xmp.size());
  return o.value_or(1);
}

// Pillow's Image.transpose for exif_transpose's methods, on an RGB image.
Rgb transpose(const Rgb& in, int orientation) {
  if (orientation < 2 || orientation > 8) return in;
  const int w = in.width, h = in.height;
  const bool swap = orientation >= 5;
  Rgb out;
  out.width = swap ? h : w;
  out.height = swap ? w : h;
  out.pixels.resize(in.pixels.size());
  for (int y = 0; y < out.height; ++y) {
    for (int x = 0; x < out.width; ++x) {
      int sx = 0, sy = 0;
      switch (orientation) {
        case 2: sx = w - 1 - x; sy = y; break;              // FLIP_LEFT_RIGHT
        case 3: sx = w - 1 - x; sy = h - 1 - y; break;      // ROTATE_180
        case 4: sx = x; sy = h - 1 - y; break;              // FLIP_TOP_BOTTOM
        case 5: sx = y; sy = x; break;                      // TRANSPOSE
        case 6: sx = y; sy = h - 1 - x; break;              // ROTATE_270 (clockwise 90)
        case 7: sx = w - 1 - y; sy = h - 1 - x; break;      // TRANSVERSE
        case 8: sx = w - 1 - y; sy = x; break;              // ROTATE_90 (counter-clockwise)
      }
      std::memcpy(&out.pixels[(std::size_t(y) * out.width + x) * 3], &in.pixels[(std::size_t(sy) * w + sx) * 3], 3);
    }
  }
  return out;
}

// ---------------------------------------------------------------- JPEG (as Pillow's JpegDecode.c drives libjpeg)

struct JpegError {
  jpeg_error_mgr pub;
  std::jmp_buf jump;
};

void jpeg_error_exit(j_common_ptr cinfo) {
  std::longjmp(reinterpret_cast<JpegError*>(cinfo->err)->jump, 1);
}
void jpeg_silent(j_common_ptr) {}

// A source that never invents data: when the bytes run out, it suspends (Pillow's buffer source, given the whole
// file at once), so a truncated image is seen as one rather than padded with a fake EOI.
struct SuspendingSource {
  jpeg_source_mgr pub;
  std::size_t skip;
};
void src_init(j_decompress_ptr) {}
boolean src_fill(j_decompress_ptr) { return FALSE; }
void src_skip(j_decompress_ptr cinfo, long n) {
  auto* s = reinterpret_cast<SuspendingSource*>(cinfo->src);
  if (n <= 0) return;
  if (std::size_t(n) > s->pub.bytes_in_buffer) {
    s->skip = std::size_t(n) - s->pub.bytes_in_buffer;
    s->pub.next_input_byte += s->pub.bytes_in_buffer;
    s->pub.bytes_in_buffer = 0;
  } else {
    s->pub.next_input_byte += n;
    s->pub.bytes_in_buffer -= std::size_t(n);
  }
}
void src_term(j_decompress_ptr) {}

// The APP1 segments libjpeg itself read before the first scan (jpeg_save_markers), as Pillow keeps them: every
// "Exif\0\0" segment concatenated (the first whole, later ones without their 6-byte header), and the last XMP packet.
// Taken from libjpeg's parse rather than a separate scan, so a damaged file's orientation comes from the same reading
// that gives its pixels (libjpeg resynchronises past a bad segment length as Pillow's marker loop does).
void jpeg_saved_metadata(const jpeg_decompress_struct& cinfo, Metadata& meta) {
  for (jpeg_saved_marker_ptr m = cinfo.marker_list; m; m = m->next) {
    if (m->marker != JPEG_APP0 + 1) continue;
    const std::uint8_t* s = m->data;
    const std::size_t sn = m->data_length;
    if (starts_with(s, sn, std::string_view("Exif\0\0", 6))) {
      if (!meta.has_exif) {
        meta.exif.assign(s, s + sn);
        meta.has_exif = true;
      } else {
        meta.exif.insert(meta.exif.end(), s + 6, s + sn);
      }
    } else if (starts_with(s, sn, std::string_view("http://ns.adobe.com/xap/1.0/\0", 29))) {
      meta.xmp.assign(s + 29, s + sn);
      meta.has_xmp = true;
    }
  }
}

void jpeg_use_source(jpeg_decompress_struct& cinfo, SuspendingSource& src, const std::uint8_t* data, std::size_t size) {
  src.pub.init_source = src_init;
  src.pub.fill_input_buffer = src_fill;
  src.pub.skip_input_data = src_skip;
  src.pub.resync_to_restart = jpeg_resync_to_restart;
  src.pub.term_source = src_term;
  src.pub.next_input_byte = data;
  src.pub.bytes_in_buffer = size;
  cinfo.src = &src.pub;
  jpeg_save_markers(&cinfo, JPEG_APP0 + 1, 0xFFFF);
}

// The orientation metadata alone (lfm::image::orientation): libjpeg's header read, nothing decoded.
void jpeg_header_metadata(const std::uint8_t* data, std::size_t size, Metadata& meta) {
  jpeg_decompress_struct cinfo{};
  JpegError err{};
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = jpeg_error_exit;
  err.pub.output_message = jpeg_silent;
  SuspendingSource src{};
  if (setjmp(err.jump)) {
    jpeg_destroy_decompress(&cinfo);
    meta = Metadata{};
    return;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_use_source(cinfo, src, data, size);
  int ok;
  do {
    ok = jpeg_read_header(&cinfo, FALSE);
  } while (ok == JPEG_HEADER_TABLES_ONLY);
  if (ok == JPEG_HEADER_OK) jpeg_saved_metadata(cinfo, meta);
  jpeg_destroy_decompress(&cinfo);
}

Rgb decode_jpeg(const std::uint8_t* data, std::size_t size, const Limits& limits, Metadata& meta) {
  jpeg_decompress_struct cinfo{};
  JpegError err{};
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.error_exit = jpeg_error_exit;
  err.pub.output_message = jpeg_silent;
  SuspendingSource src{};
  Rgb out;
  std::vector<std::uint8_t> row;
  const char* refusal = nullptr;  // set before a longjmp-free return path
  std::string refusal_message;
  if (setjmp(err.jump)) {
    char message[JMSG_LENGTH_MAX];
    err.pub.format_message(reinterpret_cast<j_common_ptr>(&cinfo), message);
    jpeg_destroy_decompress(&cinfo);
    fail("corrupt_image", std::string("JPEG: ") + message);
  }
  jpeg_create_decompress(&cinfo);
  jpeg_use_source(cinfo, src, data, size);

  int ok;
  do {
    ok = jpeg_read_header(&cinfo, FALSE);
  } while (ok == JPEG_HEADER_TABLES_ONLY);
  if (ok == JPEG_HEADER_OK) jpeg_saved_metadata(cinfo, meta);  // before the first scan, as Pillow reads them
  if (ok == JPEG_SUSPENDED) {
    refusal = "truncated_image";
    refusal_message = "JPEG: the file ends before its image header";
  } else if (cinfo.data_precision != 8) {
    refusal = "unsupported_jpeg";
    refusal_message = "JPEG: " + std::to_string(cinfo.data_precision) + "-bit samples (only 8-bit are supported)";
  } else if (cinfo.num_components != 1 && cinfo.num_components != 3 && cinfo.num_components != 4) {
    refusal = "unsupported_jpeg";
    refusal_message = "JPEG: " + std::to_string(cinfo.num_components) + " components (1, 3 or 4 are supported)";
  }
  if (!refusal) {
    try {
      check_pixels(cinfo.image_width, cinfo.image_height, limits);
    } catch (const DecodeError& e) {
      jpeg_destroy_decompress(&cinfo);
      throw;
    }
    const int comps = cinfo.num_components;
    cinfo.out_color_space = comps == 1 ? JCS_GRAYSCALE : comps == 3 ? JCS_RGB : JCS_CMYK;
    if (!jpeg_start_decompress(&cinfo)) {
      refusal = "truncated_image";
      refusal_message = "JPEG: the file ends before its image data";
    } else {
      const int w = int(cinfo.output_width), h = int(cinfo.output_height), oc = cinfo.output_components;
      out.width = w;
      out.height = h;
      out.pixels.resize(std::size_t(w) * h * 3);
      row.resize(std::size_t(w) * oc);
      for (int y = 0; y < h; ++y) {
        JSAMPROW rp = row.data();
        if (jpeg_read_scanlines(&cinfo, &rp, 1) != 1) {
          refusal = "truncated_image";
          refusal_message = "JPEG: the file ends after " + std::to_string(y) + " of " + std::to_string(h) + " rows";
          break;
        }
        std::uint8_t* o = &out.pixels[std::size_t(y) * w * 3];
        if (oc == 1) {
          for (int x = 0; x < w; ++x) o[3 * x] = o[3 * x + 1] = o[3 * x + 2] = row[x];
        } else if (oc == 3) {
          std::memcpy(o, row.data(), std::size_t(w) * 3);
        } else {  // "CMYK;I" (every byte inverted), then Pillow's cmyk2rgb
          for (int x = 0; x < w; ++x) {
            const int k = 255 - row[4 * x + 3];  // the inverted byte Pillow holds
            const int nk = 255 - k;
            for (int c = 0; c < 3; ++c) {
              const int v = 255 - row[4 * x + c];
              const int tmp = v * nk + 128;
              const int r = nk - (((tmp >> 8) + tmp) >> 8);
              o[3 * x + c] = std::uint8_t(r <= 0 ? 0 : r < 256 ? r : 255);
            }
          }
        }
      }
      // Pillow accepts a file whose rows are all there even if jpeg_finish_decompress suspends (no EOI).
      if (!refusal) jpeg_finish_decompress(&cinfo);
    }
  }
  jpeg_destroy_decompress(&cinfo);
  if (refusal) fail(refusal, refusal_message);
  return out;
}

// ---------------------------------------------------------------- PNG

std::uint32_t be32(const std::uint8_t* p) {
  return std::uint32_t(p[0]) << 24 | std::uint32_t(p[1]) << 16 | std::uint32_t(p[2]) << 8 | p[3];
}

int hex_value(std::uint8_t c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// Pillow: "".join(text.split("\n")[3:]) then bytes.fromhex (which skips whitespace between pairs).
std::optional<Bytes> raw_profile_bytes(const std::string& text) {
  std::size_t pos = 0;
  for (int k = 0; k < 3; ++k) {
    pos = text.find('\n', pos);
    if (pos == std::string::npos) return Bytes{};
    ++pos;
  }
  Bytes out;
  int hi = -1;
  for (std::size_t i = pos; i < text.size(); ++i) {
    const auto c = std::uint8_t(text[i]);
    if (c == '\n' || c == ' ' || c == '\t' || c == '\r' || c == '\v' || c == '\f') {
      if (hi >= 0) return std::nullopt;
      continue;
    }
    const int v = hex_value(c);
    if (v < 0) return std::nullopt;
    if (hi < 0) hi = v;
    else { out.push_back(std::uint8_t(hi << 4 | v)); hi = -1; }
  }
  if (hi >= 0) return std::nullopt;
  return out;
}

struct PngHeader {
  std::uint32_t width = 0, height = 0;
  int depth = 0, color = 0;
};

// CRC-32 as PNG uses it (ISO 3309; polynomial 0xEDB88320, reflected).
std::uint32_t crc32(const std::uint8_t* p, std::size_t n) {
  static const auto table = [] {
    std::array<std::uint32_t, 256> t{};
    for (std::uint32_t i = 0; i < 256; ++i) {
      std::uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
    return t;
  }();
  std::uint32_t c = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) c = table[(c ^ p[i]) & 0xFF] ^ (c >> 8);
  return c ^ 0xFFFFFFFFu;
}

void put_chunk(Bytes& out, const char* type, const std::uint8_t* d, std::size_t len) {
  const std::size_t at = out.size();
  for (int k = 3; k >= 0; --k) out.push_back(std::uint8_t(len >> (8 * k)));
  out.insert(out.end(), type, type + 4);
  out.insert(out.end(), d, d + len);
  const std::uint32_t c = crc32(out.data() + at + 4, len + 4);
  for (int k = 3; k >= 0; --k) out.push_back(std::uint8_t(c >> (8 * k)));
}

bool letters(const std::uint8_t* t) {
  for (int k = 0; k < 4; ++k)
    if (!((t[k] >= 'A' && t[k] <= 'Z') || (t[k] >= 'a' && t[k] <= 'z'))) return false;
  return true;
}

// Reads the chunks under the damage policy of validation/images-v1 (decision of 2026-11-15), collects the orientation
// metadata, and, when `clean` is given, writes the PNG stb decodes: the signature, the pixel-defining chunks (IHDR,
// PLTE, tRNS, IDAT) in file order, and IEND.
//   - A complete chunk's CRC is checked: IHDR, PLTE or IDAT with a bad one is refused; any other chunk with a bad one
//     is dropped (its metadata unused). IEND's is not needed (it carries nothing).
//   - A chunk whose length runs past the end of the file: an IDAT keeps the rest of the file as its data (the zlib
//     stream's Adler-32, checked by the patched stb, then vouches for the pixels); any other ends the reading. So does
//     a chunk type that is not four letters (the chunk structure is lost there).
//   - An unknown critical chunk is refused (the PNG specification). Anything after IEND is ignored.
PngHeader png_chunks(const std::uint8_t* p, std::size_t n, Metadata& meta, Bytes* clean) {
  PngHeader h;
  if (clean) clean->assign(p, p + 8);
  std::size_t i = 8;
  bool first = true;
  while (i + 8 <= n) {
    const std::uint32_t len = be32(p + i);
    const std::uint8_t* t = p + i + 4;
    const std::string type(reinterpret_cast<const char*>(t), 4);
    const std::uint8_t* d = p + i + 8;
    const std::size_t avail = n - i - 8;
    if (first && type == "CgBI") fail("unsupported_png", "PNG: Apple CgBI (\"iPhone\") PNGs are not supported");
    first = false;
    if (!letters(t)) break;
    if (std::uint64_t(len) + 4 > avail) {  // runs past the end of the file
      if (type == "IDAT" && clean) put_chunk(*clean, "IDAT", d, avail);
      break;
    }
    if (type == "IEND") break;
    const bool crc_ok = crc32(t, std::size_t(len) + 4) == be32(d + len);
    const bool critical = !(t[0] & 0x20);
    if (type == "IHDR" || type == "PLTE" || type == "IDAT") {
      if (!crc_ok) fail("corrupt_image", "PNG: the " + type + " chunk is damaged (its CRC does not match)");
      if (type == "IHDR" && len >= 13) {
        h.width = be32(d);
        h.height = be32(d + 4);
        h.depth = d[8];
        h.color = d[9];
      }
      if (clean) put_chunk(*clean, type.c_str(), d, len);
    } else if (critical) {
      fail(crc_ok ? "unsupported_png" : "corrupt_image",
           crc_ok ? "PNG: unknown critical chunk \"" + type + "\"" : "PNG: a critical chunk is damaged");
    } else if (!crc_ok) {
      // a damaged ancillary chunk: dropped, as libpng drops one by default
    } else if (type == "tRNS") {
      if (clean) put_chunk(*clean, "tRNS", d, len);
    } else if (type == "eXIf") {
      meta.exif.assign({'E', 'x', 'i', 'f', 0, 0});
      meta.exif.insert(meta.exif.end(), d, d + len);
      meta.has_exif = true;
    } else if (type == "iTXt") {
      // keyword \0 compression_flag compression_method language \0 translated \0 text
      const std::uint8_t* end = d + len;
      const std::uint8_t* k = std::find(d, end, 0);
      const std::string keyword(reinterpret_cast<const char*>(d), std::size_t(k - d));
      if (k + 3 <= end && k[1] == 0 && (keyword == "XML:com.adobe.xmp" || keyword == "Raw profile type exif")) {
        const std::uint8_t* lang = k + 3;
        const std::uint8_t* lend = std::find(lang, end, 0);
        const std::uint8_t* tr = lend < end ? lend + 1 : end;
        const std::uint8_t* trend = std::find(tr, end, 0);
        const std::uint8_t* text = trend < end ? trend + 1 : end;
        if (keyword == "XML:com.adobe.xmp") {
          meta.xmp.assign(text, end);
          meta.has_xmp = true;
        } else if (auto b = raw_profile_bytes(std::string(text, end))) {
          meta.raw_profile = *b;
          meta.has_raw_profile = true;
        }
      }
    } else if (type == "tEXt") {
      const std::uint8_t* end = d + len;
      const std::uint8_t* k = std::find(d, end, 0);
      if (k < end && std::string(reinterpret_cast<const char*>(d), std::size_t(k - d)) == "Raw profile type exif") {
        if (auto b = raw_profile_bytes(std::string(k + 1, end))) {
          meta.raw_profile = *b;
          meta.has_raw_profile = true;
        }
      }
    }
    i += 12 + std::size_t(len);
  }
  if (clean) put_chunk(*clean, "IEND", nullptr, 0);
  return h;
}

Rgb decode_png(const std::uint8_t* data, std::size_t size, const Limits& limits, Metadata& meta) {
  Bytes clean;
  const PngHeader h = png_chunks(data, size, meta, &clean);
  check_pixels(h.width, h.height, limits);
  if (clean.size() > std::size_t(INT32_MAX)) fail("image_too_large", "PNG: image data over 2 GiB");
  int w = 0, ht = 0, n = 0;
  Rgb out;
  // Always 4 components: stb expands gray to (g, g, g), as Pillow's L to RGB does, and the alpha is dropped. (The
  // pinned stb refuses req_comp 0, a regression upstream fixed after it; see third_party/README.md.)
  if (h.depth == 16) {
    // Every 16-bit sample v becomes round(v / 257) = (v + 128) / 257 (v / 257 never falls on .5): the nearest 8-bit
    // level. A deliberate departure from HF (Pillow saturates 16-bit gray to white and truncates the others to their
    // high byte); validation/images-v1, decision of 2026-11-15.
    stbi_us* px = stbi_load_16_from_memory(clean.data(), int(clean.size()), &w, &ht, &n, 4);
    if (!px) fail("corrupt_image", std::string("PNG: ") + stbi_failure_reason());
    out.width = w;
    out.height = ht;
    out.pixels.resize(std::size_t(w) * ht * 3);
    for (std::size_t i = 0; i < std::size_t(w) * ht; ++i)
      for (int c = 0; c < 3; ++c) out.pixels[3 * i + c] = std::uint8_t((unsigned(px[4 * i + c]) + 128) / 257);
    stbi_image_free(px);
    return out;
  }
  stbi_uc* px = stbi_load_from_memory(clean.data(), int(clean.size()), &w, &ht, &n, 4);
  if (!px) fail("corrupt_image", std::string("PNG: ") + stbi_failure_reason());
  out.width = w;
  out.height = ht;
  out.pixels.resize(std::size_t(w) * ht * 3);
  for (std::size_t i = 0; i < std::size_t(w) * ht; ++i) {
    out.pixels[3 * i] = px[4 * i];
    out.pixels[3 * i + 1] = px[4 * i + 1];
    out.pixels[3 * i + 2] = px[4 * i + 2];
  }
  stbi_image_free(px);
  return out;
}

Metadata sniff_metadata(const std::uint8_t* data, std::size_t size) {
  Metadata meta;
  if (starts_with(data, size, "\xFF\xD8\xFF")) {
    jpeg_header_metadata(data, size, meta);
  } else if (starts_with(data, size, "\x89PNG\r\n\x1A\n")) {
    try {
      png_chunks(data, size, meta, nullptr);
    } catch (const DecodeError&) {
      return Metadata{};
    }
  }
  return meta;
}

}  // namespace

Rgb load_rgb(const std::uint8_t* data, std::size_t size, const Limits& limits) {
  Metadata meta;
  Rgb img;
  if (starts_with(data, size, "\xFF\xD8\xFF")) img = decode_jpeg(data, size, limits, meta);
  else if (starts_with(data, size, "\x89PNG\r\n\x1A\n")) img = decode_png(data, size, limits, meta);
  else fail("unsupported_format", "not a JPEG or PNG image");
  return transpose(img, orientation_of(meta));
}

int orientation(const std::uint8_t* data, std::size_t size) { return orientation_of(sniff_metadata(data, size)); }

}  // namespace lfm::image
