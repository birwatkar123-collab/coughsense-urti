package dev.urti.research

import android.app.Application
import android.content.Context
import androidx.room.Room
import dev.urti.research.data.AppDatabase
import dev.urti.research.data.KaggleCnnClassifier

class UrtiApp : Application() {

    val database: AppDatabase by lazy {
        Room.databaseBuilder(this, AppDatabase::class.java, "urti_research.db")
            .fallbackToDestructiveMigration()
            .build()
    }

    val classifier: KaggleCnnClassifier by lazy { KaggleCnnClassifier.load(this) }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }

    companion object {
        lateinit var instance: UrtiApp
        fun applicationContext(): Context = instance.applicationContext
    }
}
