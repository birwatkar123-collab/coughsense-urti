package dev.urti.research.data

import androidx.room.Dao
import androidx.room.Database
import androidx.room.Delete
import androidx.room.Entity
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.RoomDatabase
import kotlinx.coroutines.flow.Flow

@Entity(tableName = "analysis_history")
data class AnalysisEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val timestampMs: Long,
    val source: String,
    val durationMs: Long,
    val score: Double,
    val positive: Boolean,
    val band: String,
)

@Dao
interface HistoryDao {
    @Insert
    suspend fun insert(entity: AnalysisEntity): Long

    @Query("SELECT * FROM analysis_history ORDER BY timestampMs DESC")
    fun observeAll(): Flow<List<AnalysisEntity>>

    @Delete
    suspend fun delete(entity: AnalysisEntity)

    @Query("DELETE FROM analysis_history")
    suspend fun clear()
}

@Database(entities = [AnalysisEntity::class], version = 1, exportSchema = false)
abstract class AppDatabase : RoomDatabase() {
    abstract fun historyDao(): HistoryDao
}