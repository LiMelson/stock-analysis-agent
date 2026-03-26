"""
股票分析助手 - 支持多轮对话

使用方式:
    python main.py
    
对话命令:
    - 输入 'exit', 'quit' 或 '退出' 结束对话
"""

import os
import logging
import warnings

# 忽略 pydub 的 ffmpeg 警告
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv", category=RuntimeWarning)

# 设置 Hugging Face 镜像源（必须在导入 model_config 之前设置！）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main():
    """主函数 - 支持多轮对话"""
    # 禁用日志输出
    logging.disable(logging.CRITICAL)
    
    # 加载环境变量
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # 检查 API 密钥
    if not os.getenv("API_KEY"):
        print("错误: 未设置 API_KEY 环境变量")
        print("请在 .env 文件中设置 API_KEY 和 BASE_URL")
        return
    
    # 初始化知识库和 RAG 工具
    print("正在初始化知识库...")
    rag_tool = None
    try:
        from configs.model_config import EmbeddingModel
        from rag.knowledge_base import KnowledgeBaseManager
        from rag.rag_retriever import RAGTool
        
        # 创建 embedding 模型
        embedding_model = EmbeddingModel()
        
        # 创建知识库管理器
        kb_manager = KnowledgeBaseManager(embedding_model)
        
        # 初始化知识库（加载已有向量存储）
        kb_manager.init_knowledge_base(
            name="默认知识库",
            description="股票分析知识库",
            vector_store_dir="vector_stores"
        )
        
        # 创建 RAG 工具
        rag_tool = RAGTool(kb_manager)
        print("知识库初始化完成！\n")
        
    except Exception as e:
        print(f"知识库初始化失败: {e}")
        print("将不使用知识库功能\n")
    
    # 【关键】只在最开始创建一次 app，复用同一个会话状态
    from core.graph import create_app
    app = create_app()
    
    print("=" * 60)
    print("  股票分析助手已启动！")
    print("  支持多轮对话，助手会记住上下文")
    print("-" * 60)
    print("  命令:")
    print("    exit / quit / 退出  - 结束对话")
    print("=" * 60)
    
    # 多轮对话循环
    round_num = 0
    while True:
        round_num += 1
        
        # 获取用户输入
        try:
            question = input(f"\n[{round_num}] 您: ").strip()
        except EOFError:
            break
        
        if not question:
            continue
        
        # 退出命令
        if question.lower() in ['exit', 'quit', '退出', 'q']:
            print("\n助手: 再见！祝您投资顺利！")
            break
        
        print(f"\n[1/3] PlanAgent: 正在分析意图和制定计划...")
        print("-" * 40)

        try:
            # 调用同一个 app，状态会自动保持（包括 chat_history）
            result = app.invoke(
                {"question": question},
                config={"configurable": {"thread_id": "cli_user"}}
            )

            print("\n[2/3] 数据获取完成，正在生成报告...")
            
            # 显示获取到的数据摘要
            sources_got = []
            if result.get("index_result"): sources_got.append("大盘指数")
            if result.get("sentiment_result"): sources_got.append("市场情绪")
            if result.get("theme_result"): sources_got.append("题材板块")
            if result.get("stock_result"): sources_got.append("个股数据")
            
            if sources_got:
                print(f"    已获取数据: {', '.join(sources_got)}")
            else:
                print("    警告: 未获取到实时数据，将基于通用框架分析")

            print("\n[3/3] SummaryAgent: 生成最终建议...")
            print("=" * 40)

            # 输出最终建议
            final_answer = result.get("final_answer", "")
            if final_answer:
                print(f"\n{final_answer}\n")
            else:
                print("\n未能生成回答\n")
                
            # 【不 break】继续下一轮对话

        except KeyboardInterrupt:
            print("\n\n助手: 对话已中断。再见！")
            break
        except Exception as e:
            print(f"\n❌ 分析失败: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
