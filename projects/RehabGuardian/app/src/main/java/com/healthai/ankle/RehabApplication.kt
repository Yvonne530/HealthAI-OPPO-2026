package com.healthai.ankle

import android.app.Application
import com.healthai.ankle.db.RehabDatabase

class RehabApplication : Application() {

    override fun onCreate() {
        super.onCreate()
        // Warm-up the Room database on a background thread
        Thread {
            RehabDatabase.getInstance(this)
        }.start()
    }
}
