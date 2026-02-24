project(
    'torchtitan',
    'cpp',
    version: run_command('cat', 'assets/version.txt', check: true).stdout().strip(),
    default_options: [
        'optimization=3',
        'warning_level=1',
        'cpp_std=c++17',
    ],
)

find = find_program('find')
res = run_command(find, 'torchtitan', '-type', 'f', '-name', '*.py', '-or', '-name', '*.safetensors')
lines = res.stdout().strip().split('\n')
sources = []
foreach f : lines
    if f != ''
        sources += f
    endif
endforeach

py = import('python').find_installation(pure: false)
pybind11_dep = dependency('pybind11', required: true)

py.install_sources(sources, preserve_path: true)

py.extension_module('helpers_cpp',
    ['torchtitan/hf_datasets/helpers.cpp'],
    install: true,
    subdir: 'torchtitan/hf_datasets',
    dependencies : [py.dependency(), pybind11_dep],
)
