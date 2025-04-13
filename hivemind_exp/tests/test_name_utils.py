from hivemind_exp.name_utils import get_name_from_peer_id, search_peer_ids_for_name

# 测试用的对等节点ID列表
TEST_PEER_IDS = [
    "QmYyQSo1c1Ym7orWxLYvCrM2EmxFTANf8wXmmE7DWjhx5N",
    "Qma9T5YraSnpRDZqRR4krcSJabThc8nwZuJV3LercPHufi",
    "Qmb8wVVVMTRmG4U1tCdaCCqietuWwpGRSbL53PA5azBViP",
]


def test_get_name_from_peer_id():
    """测试从对等节点ID生成人类可读名称的功能
    
    验证get_name_from_peer_id函数能否从对等节点ID生成一致的、人类可读的名称。
    同时测试使用下划线连接的格式选项。
    """
    # 测试标准格式的名称生成
    names = [get_name_from_peer_id(peer_id) for peer_id in TEST_PEER_IDS]
    assert names == [
        "thorny fishy meerkat",
        "singing keen cow",
        "toothy carnivorous bison",
    ]
    # 测试使用下划线连接的格式
    assert get_name_from_peer_id(TEST_PEER_IDS[-1], True) == "toothy_carnivorous_bison"


def test_search_peer_ids_for_name():
    """测试通过名称搜索对等节点ID的功能
    
    验证search_peer_ids_for_name函数能否根据人类可读名称找到对应的对等节点ID。
    包括测试不存在的名称和完全匹配的名称。
    """
    # 测试各种名称搜索情况
    names = ["none", "not an animal", "toothy carnivorous bison"]
    results = [search_peer_ids_for_name(TEST_PEER_IDS, name) for name in names]
    # 前两个名称不存在，应返回None；最后一个名称存在，应返回对应的对等节点ID
    assert results == [None, None, "Qmb8wVVVMTRmG4U1tCdaCCqietuWwpGRSbL53PA5azBViP"]
