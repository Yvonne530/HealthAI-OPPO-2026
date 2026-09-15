# MNN — keep native JNI entry points
-keep class com.alibaba.android.mnn.** { *; }
-keepclassmembers class com.alibaba.android.mnn.** { *; }

# MediaPipe
-keep class com.google.mediapipe.** { *; }

# Room — keep entity and DAO classes
-keep class com.healthai.ankle.db.** { *; }
-keepclassmembers class com.healthai.ankle.db.** { *; }

# Kotlin serialization helpers
-keepattributes *Annotation*
-keepclassmembers class kotlin.Metadata { *; }
