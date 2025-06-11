import subprocess

def run_deepseek_tests():
    result = subprocess.run(
        ["conda", "run", "-n", "DEEPSEEK_VL2_ENV", "python3", "-m", "tests.test_deepseek_vl2"],
        capture_output=True,
        text=True
    )
    
    print("STDOUT:\n", result.stdout)
    if result.returncode != 0:
        print("STDERR:\n", result.stderr)
        raise RuntimeError("Test script failed")

if __name__ == "__main__":
    run_deepseek_tests()
