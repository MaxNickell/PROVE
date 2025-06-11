from vision.deepseek_vl2 import DeepSeekVL2

def test_deepseek_vl2():
    model = DeepSeekVL2()
    print("Testing classify object")
    model.classify_object()
    
    print("Testing list objects")
    model.list_objects()
    
    print("Testing list and bound objects")
    model.list_and_bound_objects()
    
    print("Testing locate objects")
    model.locate_objects()

if __name__ == "__main__":
    test_deepseek_vl2()