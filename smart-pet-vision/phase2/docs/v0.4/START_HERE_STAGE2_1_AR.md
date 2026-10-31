# Smart Pet Vision — ONNX Stage 2.1 (غير نهائي)

هذا الأرشيف **كود وفحوص فقط**، وليس نموذجًا اجتاز ONNX Parity. **Phase 1 مغلقة** و`main` لم يُعدل. كل ملفات Drive الأصلية بقيت كما هي.

## ماذا يوجد؟

- `src/spv/`، `configs/`، `runs/manifest.json`، `tests/` و`notebooks/` نسخة موافقة لـ30/30 بصمة محفوظة من مصدر Drive المعتمد.
- `tools/onnx_parity_diagnostics.py` + `tools/run_onnx_stage2.py`: أدوات الفحص الأولى.
- `tools/onnx_repair_core.py` + `tools/run_onnx_repair.py`: بوابة قوية للفحص العددي، فصل Raw وPost-NMS، مقارنة إعدادات ONNX وMKLDNN وإصدار ONNX جديد *مرشح* بلا استبدال الأصل.
- `notebooks/00_ONNX_Repair_Verification_COLAB.ipynb`: دفتر تشغيل عملي في Colab عند توفير بيئة داعمة للبرامج المطلوبة، يتطلب صلاحية الوصول لصور الفريق الخاصة ونسخة المصدر والأوزان.
- `verification/stage2/gate_real_assets_preflight_2026-10-31/onnx_repair_matrix.json`: إثبات فحص ملفات المصدر والأوزان وعينة 50 صورة، مع حالة `BLOCKED` بسبب غياب ONNX Runtime؛ لا يوجد اختبار استدلال فعلي جديد هنا.
- `ONNX_REMEDIATION_STATUS_AR.md` و`SPV_Phase2_ONNX_Engineering_Gate_v0.3.pdf`: نتائج كاملة دون تغيير الأرقام الماضية.

## متى نعلن الإنجاز؟

**ليس الآن.** فشل A9b التاريخي `0.002014 > 0.001` ما زال قائمًا. يجب تشغيل `run_onnx_repair.py` بالأوزان الفعلية ضمن بيئة `ultralytics + onnxruntime`، وتخزين ملف JSON مستقل لكل مرشح. لا يكفي نجاح Unit Tests أو تطابق الصناديق على ثلاث صور.

الـNotebook هو أقصر طريق لإجراء هذا التشغيل على جهاز يسمح بتنزيل المكتبات، دون إعادة تدريب Phase 1 أو تغيير أوزانها. إذا لم ينجح أي مرشح، يكون الحل المهني الاحتفاظ بـPyTorch كـBackend رسمي وتوثيق NO-GO للـONNX، ثم إغلاق قرار ONNX قبل بدء عمل Multiple Animals.

## تعليمات الأمان

- الصور الخمسون وعلامات Ground Truth الخاصة **غير مرفقة** بهذا الأرشيف؛ مصدرها Drive الخاص بالفريق. لا تنشرها في GitHub.
- نسخ الأوزان وملفات ONNX الأصلية **غير مرفقة**؛ تُستخدم من Drive وفق SHA256 المثبت.
- أي تشغيل جديد يستخدم مجلد نتائج جديدًا، مع عدم استبدال `runs/` التاريخي.
- لا ترفع الحزمة إلى Drive أو GitHub باعتبارها نهائية حتى نجاح Real-Weight Gate ومراجعة الفريق؛ هذا شرط المستخدم.
