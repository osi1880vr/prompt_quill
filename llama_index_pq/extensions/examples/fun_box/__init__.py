# extensions/fun_box/__init__.py
from extension_base import PromptQuillExtension
from .new_tab import FunBoxTabExtension
from .chat_inject import ChatSillyExtension
from .ext_inject import ChatEnhancerTwistExtension

__all__ = ["FunBoxTabExtension", "ChatSillyExtension", "ChatEnhancerTwistExtension"]