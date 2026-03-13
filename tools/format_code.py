#!/usr/bin/env python3
"""
跨平台代码格式化脚本
使用 clang-format 根据 .clang-format 文件格式化 C/C++ 代码

仅使用 Python 标准库，支持 Windows 和 Linux/macOS
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 文件扩展名
SOURCE_EXTENSIONS = {'.c', '.cc', '.h', '.cpp', '.hpp', '.cxx', '.hxx'}


def which(program):
    """查找可执行文件（替代 shutil.which）"""
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    # 如果是绝对路径，直接检查
    if os.path.isabs(program) and is_exe(program):
        return program

    # 分割 PATH 环境变量
    path_env = os.environ.get('PATH', '')
    path_separator = ';' if sys.platform == 'win32' else ':'
    paths = path_env.split(path_separator)

    # 在 Windows 上，还需要检查 PATHEXT
    exe_extensions = ['']
    if sys.platform == 'win32':
        pathext = os.environ.get('PATHEXT', '.EXE;.COM;.CMD;.BAT')
        exe_extensions = pathext.split(';')

    for path in paths:
        path = path.strip('"')
        exe_file = os.path.join(path, program)

        # 先检查原始名称
        if is_exe(exe_file):
            return exe_file

        # 在 Windows 上检查带扩展名的版本
        for ext in exe_extensions:
            exe_file_ext = exe_file + ext
            if is_exe(exe_file_ext):
                return exe_file_ext

    return None


def find_clang_format():
    """查找 clang-format 可执行文件"""
    # 首先尝试从 PATH 中查找
    clang_format_names = ['clang-format', 'clang-format.exe']

    for name in clang_format_names:
        clang_format_path = which(name)
        if clang_format_path:
            return clang_format_path

    # 常见的安装路径
    common_paths = []

    if sys.platform == 'win32':
        # 展开环境变量
        localappdata = os.environ.get('LOCALAPPDATA', '')
        userprofile = os.environ.get('USERPROFILE', '')

        common_paths = [
            r'C:\Program Files\LLVM\bin\clang-format.exe',
            r'C:\Program Files (x86)\LLVM\bin\clang-format.exe',
            os.path.join(localappdata, r'Programs\LLVM\bin\clang-format.exe'),
            os.path.join(userprofile, r'scoop\apps\llvm\current\bin\clang-format.exe'),
            os.path.join(userprofile, r'.vscode\extensions\llvm-vs-code-extensions.vscode-clangd-*\bin\clang-format.exe'),
        ]
    else:
        common_paths = [
            '/usr/bin/clang-format',
            '/usr/local/bin/clang-format',
            '/opt/llvm/bin/clang-format',
            os.path.expanduser('~/.local/bin/clang-format'),
        ]

    for path in common_paths:
        # 在 Windows 上处理通配符
        if '*' in path and sys.platform == 'win32':
            import glob
            matched_paths = glob.glob(path)
            for matched_path in matched_paths:
                if os.path.isfile(matched_path) and os.access(matched_path, os.X_OK):
                    return matched_path
        elif os.path.isfile(path) and os.access(path, os.X_OK):
            return path

    return None


def find_source_files(root_dir, exclude_dirs=None):
    """递归查找所有 C/C++ 源文件"""
    if exclude_dirs is None:
        exclude_dirs = {'.git', '.svn', 'build', 'out', 'dist', 'node_modules', '.vscode', '.idea', '__pycache__'}

    source_files = []
    root_path = Path(root_dir).resolve()

    for path in root_path.rglob('*'):
        if path.is_file() and path.suffix.lower() in SOURCE_EXTENSIONS:
            # 检查是否在排除目录中
            relative_parts = path.relative_to(root_path).parts
            if not any(part in exclude_dirs for part in relative_parts):
                source_files.append(path)

    return source_files


def format_file(file_path, clang_format_path, project_root):
    """使用 clang-format 格式化单个文件"""
    try:
        # 读取原始文件内容
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            original_content = f.read()

        # 使用 clang-format 格式化
        cmd = [clang_format_path, '-style=file', str(file_path)]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )

        if result.returncode == 0:
            formatted_content = result.stdout

            # 只在内容变化时才写入
            if formatted_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(formatted_content)
                return True, str(file_path), "formatted"
            else:
                return True, str(file_path), "unchanged"
        else:
            return False, str(file_path), f"error: {result.stderr}"

    except Exception as e:
        return False, str(file_path), f"exception: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description='使用 clang-format 格式化 C/C++ 代码',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s                    # 格式化当前目录下的所有 C/C++ 文件
  %(prog)s -p ./myproject     # 格式化指定项目目录
  %(prog)s -j 4               # 使用 4 个并行线程格式化
  %(prog)s --dry-run          # 仅显示将要格式化的文件，不实际执行
        """
    )
    parser.add_argument(
        '-p', '--project',
        default='.',
        help='项目根目录 (默认: 当前目录)'
    )
    parser.add_argument(
        '-j', '--jobs',
        type=int,
        default=os.cpu_count() or 4,
        help='并行处理的线程数 (默认: CPU核心数)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='仅显示将要格式化的文件，不实际执行格式化'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='检查模式：如果有文件需要格式化，返回非零退出码'
    )
    parser.add_argument(
        '--exclude',
        nargs='+',
        default=[],
        help='要排除的目录名称'
    )

    args = parser.parse_args()

    project_root = Path(args.project).resolve()

    if not project_root.exists():
        print(f"错误: 项目目录不存在: {project_root}")
        sys.exit(1)

    # 检查 .clang-format 文件是否存在
    clang_format_file = project_root / '.clang-format'
    if not clang_format_file.exists():
        print(f"警告: 未找到 .clang-format 文件: {clang_format_file}")
        print("将使用默认的 LLVM 风格")

    # 查找源文件
    exclude_dirs = {'.git', '.svn', 'build', 'out', 'dist', 'node_modules',
                    '.vscode', '.idea', '__pycache__', 'third_party', 'external'}
    exclude_dirs.update(args.exclude)

    print(f"正在扫描项目目录: {project_root}")
    source_files = find_source_files(project_root, exclude_dirs)

    if not source_files:
        print("未找到 C/C++ 源文件")
        sys.exit(0)

    print(f"找到 {len(source_files)} 个源文件")

    if args.dry_run:
        print("\n将要格式化的文件:")
        for f in source_files:
            print(f"  {f.relative_to(project_root)}")
        sys.exit(0)

    # 查找 clang-format
    clang_format_path = find_clang_format()

    if not clang_format_path:
        print("错误: 未找到 clang-format")
        print("请确保 LLVM/Clang 已安装并添加到 PATH 环境变量中")
        print("下载地址: https://releases.llvm.org/")
        print("")
        print("常见安装方法:")
        print("  Windows: 从官网下载安装，或使用 scoop install llvm")
        print("  Ubuntu/Debian: sudo apt install clang-format")
        print("  CentOS/RHEL: sudo yum install clang-format")
        print("  macOS: brew install llvm")
        sys.exit(1)

    print(f"使用格式化工具: {clang_format_path}")
    print(f"使用 {args.jobs} 个线程并行处理\n")

    # 并行格式化文件
    formatted_count = 0
    unchanged_count = 0
    error_count = 0
    files_needing_format = []

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        future_to_file = {
            executor.submit(format_file, f, clang_format_path, project_root): f
            for f in source_files
        }

        for future in as_completed(future_to_file):
            success, file_path, message = future.result()
            rel_path = Path(file_path).relative_to(project_root)

            if success:
                if message == "formatted":
                    formatted_count += 1
                    files_needing_format.append(file_path)
                    print(f"[已格式化] {rel_path}")
                else:
                    unchanged_count += 1
                    print(f"[无需更改] {rel_path}")
            else:
                error_count += 1
                print(f"[错误] {rel_path} - {message}")

    print(f"\n{'='*50}")
    print(f"格式化完成!")
    print(f"  已格式化: {formatted_count} 个文件")
    print(f"  无需更改: {unchanged_count} 个文件")
    print(f"  错误:     {error_count} 个文件")

    if args.check and files_needing_format:
        print(f"\n检查模式: 发现 {len(files_needing_format)} 个文件需要格式化")
        sys.exit(1)

    sys.exit(0 if error_count == 0 else 1)


if __name__ == '__main__':
    main()
