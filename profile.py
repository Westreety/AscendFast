def run_profile(
    mode: ExecutionMode,
) -> ProfileResult:
    """
    对通过正确性测试的 ExecutionMode 执行性能 profile。
    
    Args:
        mode:  correctness_passed=True 的执行模式
    
    Returns:
        ProfileResult，包含优化前后延迟数据
    
    Raises:
        ValueError: 若 mode.correctness_passed != True
    """