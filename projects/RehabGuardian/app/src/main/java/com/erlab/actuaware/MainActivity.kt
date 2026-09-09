package com.erlab.actuaware

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val tv = TextView(this).apply {
            text = "Hello Android!"
            setTextSize(android.util.TypedValue.COMPLEX_UNIT_SP, 24f)
        }
        setContentView(tv)
    }
}
