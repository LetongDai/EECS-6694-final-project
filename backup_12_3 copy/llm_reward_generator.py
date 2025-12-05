import ast
import numpy as np
import inspect
from llm_config import *

# 支持多个provider
try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import google.generativeai as genai
except ImportError:
    genai = None


class LLMRewardGenerator:
    def __init__(self, api_key, model, max_tokens, provider="gemini"):
        self.provider = provider.lower()
        self.model = model
        self.max_tokens = max_tokens

        if self.provider == "anthropic":
            if anthropic is None:
                raise ImportError("anthropic package not installed. Run: pip install anthropic")
            self.client = anthropic.Anthropic(api_key=api_key)
            if not self.model:
                self.model = "claude-sonnet-4-20250514"

        elif self.provider == "gemini":
            if genai is None:
                raise ImportError("google-generativeai package not installed. Run: pip install google-generativeai")
            genai.configure(api_key=api_key)
            if not self.model:
                self.model = "gemini-1.5-flash"
            self.client = genai.GenerativeModel(self.model)

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate_reward_code(self, policy_description):
        """根据自然语言生成reward函数代码"""
        prompt = PROMPT_TEMPLATE.format(policy_description=policy_description)

        if self.provider == "anthropic":
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text

        elif self.provider == "gemini":
            response = self.client.generate_content(prompt)
            return response.text

    def validate_and_compile(self, code):
        """验证并编译生成的代码"""
        # 清理可能的markdown代码块标记
        code = code.replace("```python", "").replace("```", "").strip()

        # 安全检查
        try:
            tree = ast.parse(code)

            # 检查禁止的导入
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if isinstance(node, ast.Import):
                        module_names = [n.name for n in node.names]
                    else:
                        module_names = [node.module] if node.module else []

                    if any(f in str(module_names) for f in FORBIDDEN_IMPORTS):
                        raise ValueError(f"Forbidden import detected: {module_names}")

            # 检查函数签名
            function_found = False
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == REQUIRED_FUNCTION_NAME:
                    function_found = True

                    # 检查参数数量
                    args = node.args
                    num_required = len(args.args) - len(args.defaults)

                    if num_required != NUM_REQUIRED_PARAMS:
                        raise ValueError(
                            f"Function must have exactly {NUM_REQUIRED_PARAMS} required parameters, got {num_required}"
                        )

                    if len(args.args) != NUM_TOTAL_PARAMS:
                        raise ValueError(
                            f"Function must have {NUM_TOTAL_PARAMS} total parameters, got {len(args.args)}"
                        )

                    # 检查参数名称
                    actual_params = [arg.arg for arg in args.args]
                    if actual_params != REQUIRED_PARAMS:
                        raise ValueError(
                            f"Parameter names must be {REQUIRED_PARAMS}, got {actual_params}"
                        )

                    break

            if not function_found:
                raise ValueError(f"Function '{REQUIRED_FUNCTION_NAME}' not found in generated code")

        except SyntaxError as e:
            raise ValueError(f"Syntax error in generated code: {e}")

        # 编译执行
        namespace = {'np': np, 'numpy': np, 'Dict': dict}
        exec(code, namespace)

        if REQUIRED_FUNCTION_NAME not in namespace:
            raise ValueError(f"{REQUIRED_FUNCTION_NAME} not found in namespace after execution")

        reward_fn = namespace[REQUIRED_FUNCTION_NAME]

        # 运行时签名验证
        sig = inspect.signature(reward_fn)
        params = list(sig.parameters.values())

        if len(params) != NUM_TOTAL_PARAMS:
            raise ValueError(f"{REQUIRED_FUNCTION_NAME} must have {NUM_TOTAL_PARAMS} parameters, got {len(params)}")

        # 检查默认值
        if params[NUM_REQUIRED_PARAMS].default == inspect.Parameter.empty:
            raise ValueError(
                f"Parameter '{REQUIRED_PARAMS[NUM_REQUIRED_PARAMS]}' must have default value"
            )

        # 测试调用验证返回值
        try:
            result = reward_fn(TEST_ENV_IDX, TEST_AUCTION_RESULTS, TEST_ENV_DATA, TEST_USE_POLICY)
        except Exception as e:
            raise ValueError(f"Function failed on test input: {e}")

        # 验证返回值类型和形状
        if not isinstance(result, np.ndarray):
            raise ValueError(f"Return type must be np.ndarray, got {type(result)}")

        if result.shape != EXPECTED_RETURN_SHAPE:
            raise ValueError(f"Return shape must be {EXPECTED_RETURN_SHAPE}, got {result.shape}")

        # 验证返回值是有限数值
        if not np.all(np.isfinite(result)):
            raise ValueError(f"Return values must be finite numbers, got {result}")

        return reward_fn