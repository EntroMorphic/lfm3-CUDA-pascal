"""Host builds that link the native image front end (native/src/image_decode.cpp, image_preprocess.cpp) and the
vendored libjpeg-turbo: the HTTP harness (image input, #4) and anything else that parses image parts.

libjpeg_dir(): build/libjpeg-turbo/<machine>/ljt, built once by tools/build_libjpeg_turbo.sh.
image_link_args(compiler, flags, out_dir): the image sources compiled with the variant's compiler and sanitizer flags
(but not -Werror or -pedantic: they include the vendored stb_image), as object files, and libjpeg.a; to append to a
compile command, with "-I", str(libjpeg_dir() / "include").
"""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def libjpeg_dir():
    out = ROOT / "build" / "libjpeg-turbo" / os.uname().machine
    if not (out / "ljt" / "lib" / "libjpeg.a").exists():
        subprocess.run(["bash", str(ROOT / "tools" / "build_libjpeg_turbo.sh"), str(out)], check=True,
                       capture_output=True)
    return out / "ljt"


def image_link_args(compiler, flags, out_dir):
    ljt = libjpeg_dir()
    objects = []
    for name in ("image_decode", "image_preprocess"):
        obj = Path(out_dir) / f"{name}.o"
        subprocess.run([compiler, "-std=c++20", "-O1", "-g", *flags, "-I", str(ROOT / "native/include"),
                        "-I", str(ljt / "include"), "-c", str(ROOT / "native/src" / f"{name}.cpp"), "-o", str(obj)],
                       check=True)
        objects.append(str(obj))
    return [*objects, str(ljt / "lib" / "libjpeg.a")]
