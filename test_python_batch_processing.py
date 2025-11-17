#!/usr/bin/env python3
"""
测试Python批处理功能修复

该脚本验证YoloPose和EfficientNet的批处理功能是否正常工作。
"""

import sys
import numpy as np
from pathlib import Path

# 添加pyengine到路径
sys.path.insert(0, str(Path(__file__).parent))

from pyengine.inference.c_pipeline import YoloPosePipelineV2, EfficientNetPipelineV2
from pyengine.utils.logger import logger


def test_yolopose_batch():
    """测试YoloPose批处理功能"""
    logger.info("BatchTest", "\n" + "="*60)
    logger.info("BatchTest", "测试 YoloPose 批处理功能")
    logger.info("BatchTest", "="*60)

    # 配置参数（请根据实际情况修改）
    library_path = "./build/lib/libjetson.so"
    engine_path = "./build/yolov8n-pose.engine"  # 修改为实际路径
    max_batch_size = 4

    try:
        # 创建pipeline
        logger.info("BatchTest", f"\n1. 创建YoloPose Pipeline (max_batch_size={max_batch_size})...")
        pipeline = YoloPosePipelineV2(
            library_path=library_path,
            engine_path=engine_path,
            input_width=640,
            input_height=640,
            max_batch_size=max_batch_size,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
        pipeline.create()
        logger.info("BatchTest", "   ✓ Pipeline 创建成功")

        # 测试1：正常批处理
        logger.info("BatchTest", f"\n2. 测试批处理 (batch_size={max_batch_size})...")
        images = [
            np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            for _ in range(max_batch_size)
        ]
        results = pipeline.infer_batch(images)
        logger.info("BatchTest", f"   ✓ 批处理成功，返回 {len(results)} 个结果")

        # 测试2：单张图像
        logger.info("BatchTest", "\n3. 测试单张图像批处理...")
        results = pipeline.infer_batch([images[0]])
        logger.info("BatchTest", f"   ✓ 单张图像批处理成功，返回 {len(results)} 个结果")

        # 测试3：空列表
        logger.info("BatchTest", "\n4. 测试空列表...")
        results = pipeline.infer_batch([])
        assert results == [], "空列表应返回空结果"
        logger.info("BatchTest", "   ✓ 空列表处理正确")

        # 测试4：超过max_batch_size（应该抛出异常）
        logger.info("BatchTest", f"\n5. 测试超过max_batch_size (batch_size={max_batch_size + 2})...")
        try:
            images_large = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(max_batch_size + 2)
            ]
            results = pipeline.infer_batch(images_large)
            logger.error("BatchTest", "   ✗ 应该抛出ValueError异常")
            return False
        except ValueError as e:
            logger.info("BatchTest", f"   ✓ 正确捕获异常: {e}")

        # 测试5：内存压力测试
        logger.info("BatchTest", "\n6. 内存压力测试 (100次迭代)...")
        import gc
        for i in range(100):
            images = [
                np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
                for _ in range(max_batch_size)
            ]
            results = pipeline.infer_batch(images)
            if i % 20 == 0:
                gc.collect()
                logger.info("BatchTest", f"   迭代 {i}/100...")
        logger.info("BatchTest", "   ✓ 内存压力测试通过")

        # 关闭pipeline
        pipeline.close()
        logger.info("BatchTest", "\n✓ YoloPose 所有测试通过！")
        return True

    except FileNotFoundError:
        logger.error("BatchTest", f"\n✗ 找不到文件，请检查路径:")
        logger.error("BatchTest", f"   - library_path: {library_path}")
        logger.error("BatchTest", f"   - engine_path: {engine_path}")
        return False
    except Exception as e:
        logger.error("BatchTest", f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_efficientnet_batch():
    """测试EfficientNet批处理功能"""
    logger.info("BatchTest", "\n" + "="*60)
    logger.info("BatchTest", "测试 EfficientNet 批处理功能")
    logger.info("BatchTest", "="*60)

    # 配置参数（请根据实际情况修改）
    library_path = "./build/lib/libjetson.so"
    engine_path = "./build/feat_logits_v2.engine"  # 修改为实际路径
    max_batch_size = 8

    try:
        # 创建pipeline
        logger.info("BatchTest", f"\n1. 创建EfficientNet Pipeline (max_batch_size={max_batch_size})...")
        pipeline = EfficientNetPipelineV2(
            library_path=library_path,
            engine_path=engine_path,
            input_width=224,
            input_height=224,
            max_batch_size=max_batch_size,
            num_classes=2,
            feature_size=512
        )
        pipeline.create()
        logger.info("BatchTest", "   ✓ Pipeline 创建成功")

        # 测试1：正常批处理
        logger.info("BatchTest", f"\n2. 测试批处理 (batch_size={max_batch_size})...")
        images = [
            np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            for _ in range(max_batch_size)
        ]
        results = pipeline.infer_batch(images)
        logger.info("BatchTest", f"   ✓ 批处理成功，返回 {len(results)} 个结果")

        # 验证结果结构
        if results:
            result = results[0]
            logger.info("BatchTest", f"   结果结构: class_id={result['class_id']}, "
                  f"confidence={result['confidence']:.4f}, "
                  f"logits shape={result['logits'].shape}, "
                  f"features shape={result['features'].shape}")

        # 测试2：单张图像
        logger.info("BatchTest", "\n3. 测试单张图像批处理...")
        results = pipeline.infer_batch([images[0]])
        logger.info("BatchTest", f"   ✓ 单张图像批处理成功，返回 {len(results)} 个结果")

        # 测试3：空列表
        logger.info("BatchTest", "\n4. 测试空列表...")
        results = pipeline.infer_batch([])
        assert results == [], "空列表应返回空结果"
        logger.info("BatchTest", "   ✓ 空列表处理正确")

        # 测试4：超过max_batch_size（应该抛出异常）
        logger.info("BatchTest", f"\n5. 测试超过max_batch_size (batch_size={max_batch_size + 2})...")
        try:
            images_large = [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                for _ in range(max_batch_size + 2)
            ]
            results = pipeline.infer_batch(images_large)
            logger.error("BatchTest", "   ✗ 应该抛出ValueError异常")
            return False
        except ValueError as e:
            logger.info("BatchTest", f"   ✓ 正确捕获异常: {e}")

        # 测试5：内存压力测试
        logger.info("BatchTest", "\n6. 内存压力测试 (100次迭代)...")
        import gc
        for i in range(100):
            images = [
                np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                for _ in range(max_batch_size)
            ]
            results = pipeline.infer_batch(images)
            if i % 20 == 0:
                gc.collect()
                logger.info("BatchTest", f"   迭代 {i}/100...")
        logger.info("BatchTest", "   ✓ 内存压力测试通过")

        # 关闭pipeline
        pipeline.close()
        logger.info("BatchTest", "\n✓ EfficientNet 所有测试通过！")
        return True

    except FileNotFoundError:
        logger.error("BatchTest", f"\n✗ 找不到文件，请检查路径:")
        logger.error("BatchTest", f"   - library_path: {library_path}")
        logger.error("BatchTest", f"   - engine_path: {engine_path}")
        return False
    except Exception as e:
        logger.error("BatchTest", f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    logger.info("BatchTest", "\n" + "="*60)
    logger.info("BatchTest", "Python 批处理功能修复验证脚本")
    logger.info("BatchTest", "="*60)
    logger.info("BatchTest", "\n注意: 请确保以下文件存在:")
    logger.info("BatchTest", "  - ./build/libtrtengine_v2.so")
    logger.info("BatchTest", "  - ./models/yolov8n-pose.engine (或修改脚本中的路径)")
    logger.info("BatchTest", "  - ./models/efficientnet.engine (或修改脚本中的路径)")

    # 运行测试
    yolo_success = test_yolopose_batch()
    eff_success = test_efficientnet_batch()

    # 总结
    logger.info("BatchTest", "\n" + "="*60)
    logger.info("BatchTest", "测试总结")
    logger.info("BatchTest", "="*60)
    logger.info("BatchTest", f"YoloPose 批处理: {'✓ 通过' if yolo_success else '✗ 失败'}")
    logger.info("BatchTest", f"EfficientNet 批处理: {'✓ 通过' if eff_success else '✗ 失败'}")

    if yolo_success and eff_success:
        logger.info("BatchTest", "\n🎉 所有测试通过！批处理功能修复成功。")
        return 0
    else:
        logger.warning("BatchTest", "\n⚠️  部分测试失败，请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
