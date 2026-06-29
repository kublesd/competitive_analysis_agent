"""验证真实模型配置文件使用可移动的项目相对路径。"""

from pathlib import Path

from competitive_analysis_agent.live_config import (
    LIVE_ENV_FILE,
    PROJECT_ROOT,
)


def test_live_environment_file_is_relative_to_project_root() -> None:
    """项目移动后，默认配置仍应从当前代码仓库根目录读取。"""

    expected_root = Path(__file__).resolve().parents[1]

    assert PROJECT_ROOT == expected_root
    assert LIVE_ENV_FILE == expected_root / ".env.example"
