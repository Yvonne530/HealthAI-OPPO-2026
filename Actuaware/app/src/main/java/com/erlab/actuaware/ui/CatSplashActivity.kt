package com.healthai.ankle.ui

import android.animation.AnimatorSet
import android.animation.ObjectAnimator
import android.content.Intent
import android.graphics.Color
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.Gravity
import android.view.animation.OvershootInterpolator
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import com.healthai.ankle.R

/**
 * Splash screen shown while the MNN engine is loading.
 * Displays the fluffy cat mascot with bounce-in animation,
 * then transitions to RehabGuardianActivity after a delay.
 */
class CatSplashActivity : AppCompatActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.statusBarColor = Color.parseColor("#FFF7ED")
        window.navigationBarColor = Color.parseColor("#FFF7ED")

        // Build layout programmatically (no extra XML needed)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            gravity = Gravity.CENTER
            setBackgroundColor(Color.parseColor("#FFF7ED"))
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.MATCH_PARENT
            )
        }

        // Cat mascot image – large splash version
        val catView = ImageView(this).apply {
            setImageResource(R.drawable.ic_cat_mascot)
            layoutParams = LinearLayout.LayoutParams(320, 320).also {
                it.gravity = Gravity.CENTER_HORIZONTAL
                it.bottomMargin = 40
            }
            scaleX = 0f; scaleY = 0f; alpha = 0f
        }

        // App name
        val titleTv = TextView(this).apply {
            text = "RehabGuardian"
            textSize = 32f
            setTextColor(Color.parseColor("#F97316"))
            typeface = android.graphics.Typeface.DEFAULT_BOLD
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.bottomMargin = 12 }
            alpha = 0f
        }

        // Subtitle
        val subtitleTv = TextView(this).apply {
            text = "踝关节康复守护猫  🐾"
            textSize = 14f
            setTextColor(Color.parseColor("#C2410C"))
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            ).also { it.bottomMargin = 60 }
            alpha = 0f
        }

        // Loading dots
        val loadingTv = TextView(this).apply {
            text = "正在加载 AI 引擎…"
            textSize = 12f
            setTextColor(Color.parseColor("#78716C"))
            gravity = Gravity.CENTER
            layoutParams = LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT,
                LinearLayout.LayoutParams.WRAP_CONTENT
            )
            alpha = 0f
        }

        root.addView(catView)
        root.addView(titleTv)
        root.addView(subtitleTv)
        root.addView(loadingTv)
        setContentView(root)

        // ── Cat bounce-in animation ──
        val catScaleX = ObjectAnimator.ofFloat(catView, "scaleX", 0f, 1.15f, 0.95f, 1f)
        val catScaleY = ObjectAnimator.ofFloat(catView, "scaleY", 0f, 1.15f, 0.95f, 1f)
        val catAlpha  = ObjectAnimator.ofFloat(catView, "alpha", 0f, 1f)

        val catSet = AnimatorSet().apply {
            playTogether(catScaleX, catScaleY, catAlpha)
            duration = 700
            interpolator = OvershootInterpolator(1.4f)
        }

        val textFade = AnimatorSet().apply {
            playTogether(
                ObjectAnimator.ofFloat(titleTv,    "alpha", 0f, 1f),
                ObjectAnimator.ofFloat(subtitleTv, "alpha", 0f, 1f),
                ObjectAnimator.ofFloat(loadingTv,  "alpha", 0f, 1f)
            )
            duration = 500
            startDelay = 400
        }

        AnimatorSet().apply {
            playSequentially(catSet, textFade)
            start()
        }

        // Idle tail-wag on cat icon (subtle rotation)
        fun wag() {
            catView.animate()
                .rotation(6f).setDuration(400)
                .withEndAction {
                    catView.animate()
                        .rotation(-6f).setDuration(400)
                        .withEndAction {
                            catView.animate().rotation(0f).setDuration(300)
                                .withEndAction { Handler(Looper.getMainLooper()).postDelayed(::wag, 1200) }
                                .start()
                        }.start()
                }.start()
        }
        Handler(Looper.getMainLooper()).postDelayed(::wag, 800)

        // Transition to main after 2.2 s (models load in background)
        Handler(Looper.getMainLooper()).postDelayed({
            startActivity(Intent(this, RehabGuardianActivity::class.java))
            overridePendingTransition(android.R.anim.fade_in, android.R.anim.fade_out)
            finish()
        }, 2200)
    }
}