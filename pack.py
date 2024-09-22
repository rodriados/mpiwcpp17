#!/usr/bin/env python
"""
A thin C++17 wrapper for MPI.
@file Source code packing script.
@author Rodrigo Siqueira <rodriados@gmail.com>
@copyright 2024-present Rodrigo Siqueira
"""
import re, os

from argparse import ArgumentParser
from configparser import ConfigParser
from dataclasses import dataclass
from typing import TextIO

cpp_preprocessor_include_regex = re.compile(rf'^#include *[<"](.*)([>"])$', re.MULTILINE)
cpp_preprocessor_pragma_once_regex = re.compile(r'^#pragma once$', re.MULTILINE)
cpp_header_comment_regex = re.compile(r'^/\*\*.*?\*/', re.DOTALL)

source_cleaning_regexs = [
    cpp_header_comment_regex
  , cpp_preprocessor_pragma_once_regex
  , cpp_preprocessor_include_regex
]

# The information of the project to be packed. Concretely, this type informs which
# directory the project's source code is located on, what is its namespace and which
# file is its entrypoint.
# @since 1.0
@dataclass
class ProjectInfo:
    workingdir: str
    namespace: str
    entrypoint: str

# The include dependency graph for a project file. This type separates what dependencies
# are language-globals and which are project files that must be packed.
# @since 1.0
@dataclass
class IncludeGraph:
    language: set[str]
    project: dict[str, list[str]]

# Creates a graph with the include-dependencies of the project's source code files.
# @param filepath The project's file to create an include graph from.
# @param project The project's information instance
# @return The include dependencies graph for the given file.
def find_include_graph(filepath: str, project: ProjectInfo) -> IncludeGraph:
    source_code_language_includes = set()
    source_code_include_graph = dict()
    unprocessed_include_files = [filepath]

    while unprocessed_include_files:
        current_file = unprocessed_include_files.pop(0)

        [project_include_files, language_include_files] = \
            extract_include_dependencies(current_file, project)

        for header in project_include_files:
            if header not in source_code_include_graph:
                unprocessed_include_files.append(header)

        source_code_include_graph[current_file] = project_include_files
        source_code_language_includes |= set(language_include_files)

    return IncludeGraph(
        project = source_code_include_graph
      , language = source_code_language_includes)

# Extracts the include-dependencies of a project file.
# @param filename The file to be analyzed and have its dependencies extracted.
# @param project The project's information instance.
# @return A tuple with the file's language-global and project dependencies.
def extract_include_dependencies(filename: str, project: ProjectInfo) -> tuple:
    with open(filename, 'r') as fhandle:
        source_code = fhandle.read()

    language_include_files = []
    project_include_files = []

    for match in re.finditer(cpp_preprocessor_include_regex, source_code):
        if match.group(2) == '"':
            current_directory = os.path.dirname(filename)
            source_file_path = os.path.join(current_directory, match.group(1))
            project_include_files.append(os.path.abspath(source_file_path))

        elif project.namespace in match.group(1):
            source_file_path = os.path.join(project.workingdir, match.group(1))
            project_include_files.append(os.path.abspath(source_file_path))

        else: language_include_files.append(match.group(1))

    return (project_include_files, language_include_files)

# Iterates over the graph and finds the required include order from the given file.
# @param srcfile The file to iterate over the graph as an entrypoint.
# @param visited The list of source files that have been already visited.
# @param graph The project's include graph instance.
# @return The require include order from the given entrypoint.
def find_graph_required_order(srcfile: str, graph: IncludeGraph, visited: list[str]) -> list[str]:
    if srcfile in visited:
        return []

    source_dependencies = []
    visited = visited + [srcfile]

    for dependency in graph.project[srcfile]:
        transitive_dependencies = [] if dependency in source_dependencies \
            else find_graph_required_order(dependency, graph, visited)

        for transitive_dependency in transitive_dependencies:
            if transitive_dependency not in source_dependencies:
                source_dependencies.append(transitive_dependency)

    return source_dependencies + [srcfile]

# Copies the contents of a given source file to an output file.
# @param srcfile The path of source file to have its contents copied.
# @param outfhandle The target output file handle to copy source to.
def copy_code_to_file(srcfile: str, outfhandle: TextIO) -> None:
    with open(srcfile, 'r') as fhandle:
        source_contents = fhandle.read()

    for regex in source_cleaning_regexs:
        source_contents = re.sub(regex, str(), source_contents)

    source_contents = '\n'.join([line for line in source_contents.splitlines() if line])

    print(source_contents, file = outfhandle)

# Copies the whole project's source code into a compacted file.
# @param outfile The target file to be written with the project source code.
# @param graph The project files' include-dependency graph.
# @param project The project information instance.
def write_compacted_source_code(outfile: str, graph: IncludeGraph, project: ProjectInfo) -> None:
    namespace = project.namespace.upper()
    include_order = find_graph_required_order(project.entrypoint, graph, [])

    with open(project.entrypoint, 'r') as fhandle:
        entrypoint_header = re                               \
            .match(cpp_header_comment_regex, fhandle.read()) \
            .group(0)

    with open(outfile, 'w') as fhandle:
        print(entrypoint_header, file = fhandle)

        print(f'#ifndef {namespace}_HEADER_INCLUDED', file = fhandle)
        print(f'#define {namespace}_HEADER_INCLUDED', file = fhandle)

        for header in graph.language:
            print(f'#include <{header}>', file = fhandle)

        for include_file_path in include_order:
            copy_code_to_file(include_file_path, fhandle)

        print(f'#endif //{namespace}_HEADER_INCLUDED', file = fhandle)

# Compacts the source code of the whole project into a single file.
# @param outfile The target file to compact the project's source code to.
# @param project The project information instance.
def compact_source_code(*, outfile: str, project: ProjectInfo) -> None:
    destination_file_path = os.path.abspath(outfile)
    entrypoint_file_path = os.path.abspath(
        os.path.join(project.workingdir, project.entrypoint))
    project_include_graph = find_include_graph(entrypoint_file_path, project)

    write_compacted_source_code(
        outfile = destination_file_path
      , graph = project_include_graph
      , project = ProjectInfo(
            workingdir = project.workingdir
          , namespace = project.namespace
          , entrypoint = entrypoint_file_path))

if __name__ == '__main__':
    config = ConfigParser()
    parser = ArgumentParser(description = "C++ source code packing script")

    parser.add_argument('-c', '--config',
        help = 'The source packing script configuration file',
        nargs = '?', metavar = 'file', dest = 'config', default = '.packconfig')

    parser.add_argument('-o', '--outfile',
        help = 'The target file to output the packed source code to',
        nargs = '?', metavar = 'file', dest = 'outfile')

    args = parser.parse_args()
    config.read(args.config)

    compact_source_code(
        outfile = config['output']['outfile'] if args.outfile is None else args.outfile,
        project = ProjectInfo(
            workingdir = config['project']['workingdir']
          , namespace = config['project']['namespace']
          , entrypoint = config['project']['entrypoint']))
