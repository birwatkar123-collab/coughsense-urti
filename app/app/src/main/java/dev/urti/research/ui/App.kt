package dev.urti.research.ui

import androidx.compose.runtime.Composable
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import dev.urti.research.ui.screens.AboutScreen
import dev.urti.research.ui.screens.HistoryScreen
import dev.urti.research.ui.screens.HomeScreen
import dev.urti.research.ui.screens.ImportScreen
import dev.urti.research.ui.screens.RecordScreen
import dev.urti.research.ui.screens.AnalyzeScreen

object Routes {
    const val HOME = "home"
    const val RECORD = "record"
    const val IMPORT = "import"
    const val ANALYZE = "analyze"
    const val HISTORY = "history"
    const val ABOUT = "about"
}

@Composable
fun App(navController: NavHostController = rememberNavController()) {
    val vm: AnalysisViewModel = viewModel()

    NavHost(navController = navController, startDestination = Routes.HOME) {
        composable(Routes.HOME) { HomeScreen(onNavigate = navController::navigate) }
        composable(Routes.RECORD) {
            RecordScreen(vm = vm, onBack = { navController.popBackStack() },
                onAnalyzing = { navController.navigate(Routes.ANALYZE) { launchSingleTop = true } })
        }
        composable(Routes.IMPORT) {
            ImportScreen(vm = vm, onBack = { navController.popBackStack() },
                onAnalyzing = { navController.navigate(Routes.ANALYZE) { launchSingleTop = true } })
        }
        composable(Routes.ANALYZE) {
            AnalyzeScreen(
                vm = vm,
                onHome = {
                    vm.reset()
                    navController.navigate(Routes.HOME) { popUpTo(Routes.HOME) { inclusive = true } }
                },
                onBack = { vm.reset(); navController.popBackStack() },
            )
        }
        composable(Routes.HISTORY) {
            HistoryScreen(vm = vm, onBack = { navController.popBackStack() })
        }
        composable(Routes.ABOUT) {
            AboutScreen(metadata = vm.classifierMetadata(), onBack = { navController.popBackStack() })
        }
    }
}