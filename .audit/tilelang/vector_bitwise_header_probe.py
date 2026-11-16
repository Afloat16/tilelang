import json
import subprocess
from pathlib import Path

repo = Path('/workspace/scratch/4bed741d4966/repos/tilelang')
out = Path('/workspace/scratch/4bed741d4966/audit/tilelang/vector-header-probe')
out.mkdir(exist_ok=True)
operators = {'and': 'a & b', 'or': 'a | b', 'xor': 'a ^ b', 'not': '~a', 'left': 'a << b', 'right': 'a >> b', 'add_control': 'a + b'}
results = []
for name, expr in operators.items():
    source = '#include "tl_templates/cpp/common.h"\nint main() {\n'
    source += '  int32_t4 a(8), b(1);\n'
    source += f'  auto c = {expr};\n'
    source += '  return c.data[0] == 0x7fffffff;\n}\n'
    path = out / f'{name}.cc'
    path.write_text(source)
    command = ['g++', '-std=c++17', '-I', str(repo / 'src'), str(path), '-o', str(out / name)]
    proc = subprocess.run(command, text=True, capture_output=True)
    (out / f'{name}.log').write_text(proc.stdout + proc.stderr)
    results.append({'operator': name, 'command': command, 'compile_returncode': proc.returncode, 'diagnostic': proc.stderr[:1600]})
report = {'base': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(), 'scope': 'native template-header compilation only; public TileLang kernel not yet exercised', 'results': results}
(out / 'baseline.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'base': report['base'], 'results': [{k: v for k, v in row.items() if k != 'command'} for row in results]}, indent=2))
