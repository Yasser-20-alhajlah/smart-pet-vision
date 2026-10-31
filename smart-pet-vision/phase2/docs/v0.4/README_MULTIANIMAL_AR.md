# Smart Pet Vision - Phase 2 / Multiple Animals Review Candidate v0.4

**الحالة:** إصلاح برمجي تجريبي موثّق؛ لم يُعتمد بعد على الأوزان الحقيقية أو بيانات اختبار جديدة. **نطاق العمل:** Phase 2 فقط، دون المساس بملفات Phase 1 أو تدريب النماذج أو أوزانها أو المصدر الرئيسي.

## ماذا تغير ولماذا؟

المصدر المعتمد 30 أكتوبر 2026 يحيل الصورة للمراجعة إذا وجد أكثر من حيوان بواسطة Detector أو Segmenter. لكن سجل تشغيل الفريق على 401 صورة سبق فحصها يظهر استمرار **17 من أصل 52** صورة متعددة الحيوانات في حالة Ready المبدئية. لا يعني هذا وجود موافقة نشر؛ الموافقة البشرية مطلوبة دائمًا.

أضفنا مرحلة **Secondary Animal Review** بعد Pipeline الأساسي، ولا تُشغَّل إلا إذا كانت النتيجة ستصبح `ready`. تعيد فحص الصورة بواسطة نفس الأوزان الأصلية بدقة صورة أعلى وثقة كشف أقل. يمكن إضافة كاشف Segmenter مستقل على الدقة الأعلى، ويدعم الكود عرضًا اختياريًا من أربع Tiles متداخلة للحيوانات الصغيرة. نقارن Bounding Boxes داخل إحداثيات الصورة الأصلية ونزيل تكرارات الكائن نفسه قدر الإمكان. أي Candidate إضافي أو ملتبس يؤدي إلى `refer_to_staff`، **وليس إثبات عدد الحيوانات**.

## سياسة التشغيل الآمن

- `review_mode: off` - يحافظ على سلوك المصدر التاريخي.
- `review_mode: audit` - يسجل المرشحات والأزمنة ولا يغيّر القرار؛ للتطوير فقط.
- `review_mode: enforce` - يحيل الصور المؤهلة إلى مراجعة بشرية عند وجود مرشحات جديدة أو فشل Secondary Review.
- `review_tiles_enabled: false` مبدئيًا؛ ملف منفصل يفعّل Tiles بوصفه تجربة High Recall، وليس اختيارًا مثبتًا من نتائج الـ401.
- أي وضع لا يسمح بالنشر الآلي، ولا يزيل موافقة الموظف ولا يغيّر نتائج Cat/Dog أو Masks.

## ملفات التعديل

| الملف | التغيير |
|---|---|
| `src/spv/pipeline.py` | Manifest controls، Secondary Review، سبب الإحالة، سجل منفصل لتوقيت الفحص وأدلته |
| `src/spv/multi_animal_review.py` | فحص Boxes/Confidence، مقارنة IoU، دمج المقترحات، إحداثيات Tiles |
| `tools/evaluate_multi_animal_candidate.py` | مقارنة Baseline وCandidate في تشغيل واحد، يرفض أي نقص في الصور أو سجلات الحقيقة |
| `tests/test_multi_animal_review.py` | 16 اختبارًا للأخطاء، حالات الحافة، Security/Fallback، المعالجة المرحلية |
| `tests/test_multi_animal_evaluator.py` | اختباران لاتساق البيانات وكلفة الإحالة والمقارنة المزدوجة |
| `runs/manifest_multi_animal_candidate.json` | نسخة إعدادات تجريبية Enforce مع High-Resolution وSegmenter |
| `runs/manifest_multi_animal_tiled_experimental.json` | إصدار تجريبي منفصل يضيف فحص Tiles |
| `.gitignore` | السماح بتتبع ملفات Manifest المرشحة دون لمس `runs/manifest.json` |

## كيفية إعادة التشغيل بالأوزان الحقيقية

تحقق من وجود ملفات `detector.pt`, `segmenter.pt`, `classifier.pt` في المسارات المحددة بملف Manifest، ولا تغير أوزان التشغيل الأصلية. شغّل:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
PYTHONPATH=src python tools/evaluate_multi_animal_candidate.py \
  --manifest runs/manifest_multi_animal_candidate.json \
  --images /PRIVATE/SmartPetVision_TeamData/images \
  --labels /PRIVATE/SmartPetVision_TeamData/labels.csv \
  --output review_runs/multi_animal_v04_new_run
```

**تنبيه:** هذا الأمر يحتاج `torch`, `ultralytics` والمكتبات الأخرى في بيئة الفريق. مجلد `output` جديد حصريًا؛ الأداة ترفض استبدال النتائج السابقة. مجموعة الـ401 هي **Regression** وليست Untouched Holdout، ولا يجوز اختيار Conf أو IoU بناءً عليها. للتعميم يلزم مجموعة Development منفصلة لتحديد المعايير ثم Holdout جديد لم يُستخدم خلال التطوير، مع قياس False-ready وSingle-animal Referrals وLatency.

## شروط الاعتماد النهائي

1. إعادة تشغيل المصدر الحالي مع الأوزان الموجودة، وحفظ Hash لكل ملف وبيئة التشغيل.
2. مقارنتان من التشغيل نفسه: `baseline_verdict` و`candidate_verdict` لكل صورة.
3. تسجيل نسبة `multi_animal_false_ready_rate` **مع** زيادة `single_animal_referral_rate`، لا نكتفي برقم واحد.
4. اختبار مستقل جديد غير مستخدم للتعديل واختبار T4 Pipeline p95 بالنسبة لمتطلبات المشروع.
5. لا دمج إلى GitHub `main` ولا تسمية المشروع Release نهائيًا قبل اجتياز هذه الخطوات.

**خارج النطاق:** أي تغييرات على Component Export/ONNX أو أوزان النماذج أو نتائج Phase 1. سجلات التحقق القديمة تبقى موجودة دون تعديل.
