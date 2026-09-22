package dev.urti.research

import android.app.Application
import android.content.Context
import androidx.room.Room
import dev.urti.research.data.AppDatabase
import dev.urti.research.data.V4Classifier
import dev.urti.research.data.YamnetEmbedder

class UrtiApp : Application() {

    val database: AppDatabase by lazy {
        Room.databaseBuilder(this, AppDatabase::class.java, "urti_research.db")
            .fallbackToDestructiveMigration()
            .build()
    }

    val embedder: YamnetEmbedder by lazy { YamnetEmbedder.load(this) }
    val classifier: V4Classifier by lazy { V4Classifier.load(this) }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: UrtiApp
        fun applicationContext(): Context = instance.applicationContext
    }
}