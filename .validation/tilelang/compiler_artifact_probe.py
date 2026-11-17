"""Measure real CPU compiler failure artifacts without changing product source."""
import hashlib
import json
import os
import pathlib
import subprocess
import tempfile
from tilelang import tvm
from tilelang.env import env
from tilelang.jit.adapter.libgen import LibraryGenerator

def main():
    rows=[]
    original_tempdir = tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix="tilelang-compiler-failure-") as work:
        tempfile.tempdir=work
        try:
            for index in range(3):
                generator=LibraryGenerator(tvm.target.Target("c"))
                generator.assign_pass_configs({})
                generator.update_lib_code('extern "C" int answer() { return missing_identifier; }\n')
                try:
                    generator.compile_lib()
                except RuntimeError as exc:
                    rows.append({"case":"real_compile_failure","iteration":index,"exception":str(exc).splitlines()[0],"files_after":[p.name for p in pathlib.Path(work).iterdir()],"generator_source_path":generator.get_source_path(),"generator_library_path":generator.get_lib_path()})
                else:
                    raise AssertionError("Invalid C++ unexpectedly compiled")
            generator=LibraryGenerator(tvm.target.Target("c"))
            generator.assign_pass_configs({})
            generator.update_lib_code('extern "C" int answer() { return 42; }\n')
            generator.compile_lib()
            library=generator.load_lib()
            result=library.answer()
            assert result==42
            rows.append({"case":"successful_compile_control","answer":result,"source_retained":pathlib.Path(generator.get_source_path()).is_file(),"library_retained":pathlib.Path(generator.get_lib_path()).is_file()})
        finally:
            tempfile.tempdir=original_tempdir
    output=pathlib.Path(__file__).parent/"reports"/"compiler-artifacts.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    report={"source_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"libgen_sha256":hashlib.sha256(pathlib.Path("tilelang/jit/adapter/libgen.py").read_bytes()).hexdigest(),"cleanup_policy":env.should_cleanup_temp_files(),"rows":rows}
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)
if __name__=="__main__":
    main()
