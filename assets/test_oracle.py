

def test_remove_repeated_paths_2():
    """ """
    paths = [
        [1],
        [1,3],
        [1],
        [1,2],
        [1,3]
    ]

    nondup_paths = remove_repeated_paths(paths)
    assert [[1], [1, 3], [1, 2]] == nondup_paths



def test_remove_repeated_paths_1():
    """ """
    paths = [
        [1],
        [1],
        [1],
        [1]
    ]

    nondup_paths = remove_repeated_paths(paths)
    assert nondup_paths == [[1]]



def test_remove_repeated_paths_3():
    """ """
    paths = [
        [1],
        [1,2],
        [1,3],
        [1,4]
    ]
    nondup_paths = remove_repeated_paths(paths)
    assert nondup_paths == [[1],[1,2],[1,3],[1,4]]




    #bfs_failure()
    #dfs_recurs_test()
    # dfs_iterative_test_1()
    # dfs_iterative_test_2() 
    # dfs_iterative_test_3() 

        test_remove_repeated_paths_1()
    test_remove_repeated_paths_2()
    test_remove_repeated_paths_3()

def dfs_iterative_test_1():
    """
    """
    start = 'A'

    graph = {
        'A' : ['B','C'],
        'B' : ['D','E'],
        'C' : ['D','E']
    }

    len_2_paths = [
        ['A', 'B'], 
        ['A', 'C']
    ]

    len_3_paths = [
        ['A', 'B'], 
        ['A', 'C'],
        ['A', 'C', 'D'],
        ['A', 'C', 'E'],
        ['A', 'B', 'D'],
        ['A', 'B', 'E']
    ]

    import pdb
    # 
    paths = find_all_paths_from_src(graph, start, max_depth=1, remove_duplicates=False)
    pdb.set_trace()
    assert paths == len_2_paths
    paths = find_all_paths_from_src(graph, start, max_depth=2, remove_duplicates=False)
    pdb.set_trace()
    assert paths == len_3_paths



def get_sample_graph() -> Mapping[str, List[str]]:
    """
        Args:
            None

        Returns:
            graph: Python dictionary representing an adjacency list
    """
    graph = {"1": ["2", "3", "4"], "2": ["5", "6"], "5": ["9", "10"], "4": ["7", "8"], "7": ["11", "12"]}
    return graph




def dfs_iterative_test_2() -> None:
    """Graph is in adjacent list representation."""
    graph = get_sample_graph()
    paths_ref_depth3 = [
        ["1", "3"],
        ["1", "2", "6"],
        ["1", "4", "8"],
        ["1", "2", "5", "9"],
        ["1", "2", "5", "10"],
        ["1", "4", "7", "11"],
        ["1", "4", "7", "12"],
    ]

    paths = find_all_paths_from_src(graph, start="1", max_depth=3)
    import pdb
    pdb.set_trace()
    #compare_sets_of_lists( paths_ref_depth3, paths) 
    ref_paths = [['1', '3'], ['1', '4', '8'], ['1', '4', '7', '11'], ['1', '4', '7', '12'], ['1', '2', '6'], ['1', '2', '5', '9'], ['1', '2', '5', '10']]
    assert paths == ref_paths


def dfs_iterative_test_3() -> None:
    """Graph is in adjacent list representation."""
    graph = get_sample_graph()
    paths_ref_depth2 = [["1", "3"], ["1", "2", "6"], ["1", "4", "8"], ["1", "2", "5"], ["1", "4", "7"]]
    paths = find_all_paths_from_src(graph, start="1", max_depth=2)
    # pdb.set_trace()
    compare_sets_of_lists( paths_ref_depth2, paths) 
    assert paths == [['1', '3'], ['1', '4', '7'], ['1', '4', '8'], ['1', '2', '5'], ['1', '2', '6']]


def compare_sets_of_lists(set1, set2):
    """
    """
    assert len(set1) == len(set2)

    print(set1)
    print(set2)

    for list_item in set1:
        print('\t',list_item)
        assert list_item in set2

    for list_item in set2:
        print('\t',list_item)
        assert list_item in set1


def bfs_failure():
    """ """
    start_id = 'A'

    graph = {
        'A' : ['B','C'],
        'B' : ['D','E'],
        'C' : ['D','E']
    }
    import pdb
    pdb.set_trace()
    paths = bfs_enumerate_paths(graph, str(start_id), max_depth=BFS_MAX_DEPTH)
    print(paths)


def dfs_recurs_test():
    """
    """
    start = 'A'

    graph = {
        'A' : ['B','C'],
        'B' : ['D','E'],
        'C' : ['D','E']
    }
    import pdb
    #pdb.set_trace()
    paths = find_all_paths_from_src_recursive(graph, start, max_depth=2)
    print(paths)
    paths = find_all_paths_from_src_recursive(graph, start, max_depth=3)
    print(paths)


def find_all_paths_from_src_recurs(graph, start, max_depth, path=[]):
    """ 
    from source only, recursively

    """
    path = path + [start]
    if (not start in graph) or (len(path)+1 > max_depth):
        return [path]
    paths = [path]
    for node in graph[start]:
        if node not in path:
            paths.extend( find_all_paths_from_src_recurs(graph, node, max_depth, path) )

    return paths



