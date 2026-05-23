from typing import Optional, List, Dict

from langchain_core.tools import BaseTool


def get_tools(tool_names: Optional[List[str]] = None, include_gmail: bool = False) -> List[BaseTool]:
    """Get specified tools or all tools if tool_names is None.

    Args:
        tool_names: Optional list of tool names to include. If None, returns all tools.
        include_gmail: Whether to include Gmail tools. Defaults to False.

    Returns:
        List of tool objects
    """
    # Import default tools
    from tools import write_an_article, schedule_date_for_article, check_daily_number_article

    # Base tools dictionary
    all_tools = {
        "write_article": write_an_article,
        "schedule_date_article": schedule_date_for_article,
        "check_calendar_availability": check_daily_number_article,
    }
    if tool_names is None:
        return list(all_tools.values())

    return [all_tools[name] for name in tool_names if name in all_tools]


def get_tools_by_name(tools: Optional[List[BaseTool]] = None) -> Dict[str, BaseTool]:
    """Get a dictionary of tools mapped by name."""
    if tools is None:
        tools = get_tools()

    return {tool.name: tool for tool in tools}