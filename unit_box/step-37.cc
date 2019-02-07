/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2009 - 2018 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Authors: Katharina Kormann, Martin Kronbichler, Uppsala University,
 * 2009-2012, updated to MPI version with parallel vectors in 2016
 */


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <iostream>
#include <fstream>


namespace Step37
{
using namespace dealii;


const unsigned int degree_finite_element = 1;
const unsigned int dimension = 2;



template <int dim>
class Coefficient : public Function<dim>
{
public:
    Coefficient ()  : Function<dim>() {}

    virtual double value (const Point<dim>   &p,
                          const unsigned int  component = 0) const;

    template <typename number>
    number value (const Point<dim,number> &p,
                  const unsigned int component = 0) const;

    virtual void value_list (const std::vector<Point<dim> > &points,
                             std::vector<double>            &values,
                             const unsigned int              component = 0) const;
};



template <int dim>
template <typename number>
number Coefficient<dim>::value (const Point<dim,number> &p,
                                const unsigned int /*component*/) const
{
    return p.square()/p.square(); //1. / (0.05 + 2.*p.square());
}



template <int dim>
double Coefficient<dim>::value (const Point<dim>  &p,
                                const unsigned int component) const
{
    return value<double>(p,component);
}



template <int dim>
void Coefficient<dim>::value_list (const std::vector<Point<dim> > &points,
                                   std::vector<double>            &values,
                                   const unsigned int              component) const
{
    Assert (values.size() == points.size(),
            ExcDimensionMismatch (values.size(), points.size()));
    Assert (component == 0,
            ExcIndexRange (component, 0, 1));

    const unsigned int n_points = points.size();
    for (unsigned int i=0; i<n_points; ++i)
        values[i] = value<double>(points[i],component);
}



template <int dim>
class RightHandSide : public Function<dim>
{
public:
    RightHandSide ()  : Function<dim>(dim) {}

    virtual void vector_value (const Point<dim> &p,
                               Vector<double>   &values) const;
};



template <int dim>
void RightHandSide<dim>::vector_value (const Point<dim> &p,
                                       Vector<double>   &values) const
{
    (void)p;

    for (unsigned int i=0; i<values.size(); ++i)
        values(i) = 1.0;

    return ;
}




template <int dim, int fe_degree, typename number>
class LaplaceOperator : public MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >
{
public:
    typedef number value_type;

    LaplaceOperator ();

    void clear();

    void evaluate_coefficient(const Coefficient<dim> &coefficient_function);

    Table<2, VectorizedArray<number> > get_coefficient_table ();

    virtual void compute_diagonal();

private:
    virtual void apply_add(LinearAlgebra::distributed::Vector<number> &dst,
                           const LinearAlgebra::distributed::Vector<number> &src) const;

    void local_apply (const MatrixFree<dim,number>                     &data,
                      LinearAlgebra::distributed::Vector<number>       &dst,
                      const LinearAlgebra::distributed::Vector<number> &src,
                      const std::pair<unsigned int,unsigned int>       &cell_range) const;

    void local_compute_diagonal (const MatrixFree<dim,number>                     &data,
                                 LinearAlgebra::distributed::Vector<number>       &dst,
                                 const unsigned int                               &dummy,
                                 const std::pair<unsigned int,unsigned int>       &cell_range) const;

    Table<2, VectorizedArray<number> > coefficient;
};



template <int dim, int fe_degree, typename number>
LaplaceOperator<dim,fe_degree,number>::LaplaceOperator ()
    :
      MatrixFreeOperators::Base<dim, LinearAlgebra::distributed::Vector<number> >()
{}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::clear ()
{
    coefficient.reinit(0, 0);
    MatrixFreeOperators::Base<dim,LinearAlgebra::distributed::Vector<number> >::clear();
}




template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>::
evaluate_coefficient (const Coefficient<dim> &coefficient_function)
{
    const unsigned int n_cells = this->data->n_macro_cells();
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,number> phi (*this->data);

    coefficient.reinit (n_cells, phi.n_q_points);
    for (unsigned int cell=0; cell<n_cells; ++cell)
    {
        phi.reinit (cell);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
            coefficient(cell,q) =
                    coefficient_function.value(phi.quadrature_point(q));
    }
}

template <int dim, int fe_degree, typename number>
Table<2, VectorizedArray<number> >
LaplaceOperator<dim,fe_degree,number>::
get_coefficient_table()
{
    return coefficient;
}


template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>
::local_apply (const MatrixFree<dim,number>                     &data,
               LinearAlgebra::distributed::Vector<number>       &dst,
               const LinearAlgebra::distributed::Vector<number> &src,
               const std::pair<unsigned int,unsigned int>       &cell_range) const
{
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,number> phi (data);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
        AssertDimension(coefficient.size(0), data.n_macro_cells());
        AssertDimension(coefficient.size(1), phi.n_q_points);

        phi.reinit (cell);
        phi.read_dof_values(src);
        phi.evaluate (false, true);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
            phi.submit_gradient (coefficient(cell,q) *
                                 phi.get_gradient(q), q);
        phi.integrate (false, true);
        phi.distribute_local_to_global (dst);
    }
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>
::apply_add (LinearAlgebra::distributed::Vector<number>       &dst,
             const LinearAlgebra::distributed::Vector<number> &src) const
{
    this->data->cell_loop (&LaplaceOperator::local_apply, this, dst, src);
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>
::compute_diagonal ()
{
    this->inverse_diagonal_entries.
            reset(new DiagonalMatrix<LinearAlgebra::distributed::Vector<number> >());
    LinearAlgebra::distributed::Vector<number> &inverse_diagonal =
            this->inverse_diagonal_entries->get_vector();
    this->data->initialize_dof_vector(inverse_diagonal);
    unsigned int dummy = 0;
    this->data->cell_loop (&LaplaceOperator::local_compute_diagonal, this,
                           inverse_diagonal, dummy);

    this->set_constrained_entries_to_one(inverse_diagonal);

    for (unsigned int i=0; i<inverse_diagonal.local_size(); ++i)
    {
        Assert(inverse_diagonal.local_element(i) > 0.,
               ExcMessage("No diagonal entry in a positive definite operator "
                          "should be zero"));
        inverse_diagonal.local_element(i) =
                1./inverse_diagonal.local_element(i);
    }
}



template <int dim, int fe_degree, typename number>
void
LaplaceOperator<dim,fe_degree,number>
::local_compute_diagonal (const MatrixFree<dim,number>               &data,
                          LinearAlgebra::distributed::Vector<number> &dst,
                          const unsigned int &,
                          const std::pair<unsigned int,unsigned int> &cell_range) const
{
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,number> phi (data);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
        phi.reinit (cell);
        AlignedVector<VectorizedArray<number> > diagonal(phi.dofs_per_cell);
        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
        {
            for (unsigned int j=0; j<phi.dofs_per_cell; ++j)
                phi.begin_dof_values()[j] = make_vectorized_array<number> (0.0);
            phi.begin_dof_values()[i] = make_vectorized_array<number> (1.);

            phi.evaluate (false,true,false);
            for (unsigned int q=0; q<phi.n_q_points; ++q)
                phi.submit_gradient(coefficient(cell,q)*phi.get_gradient(q), q);

            phi.integrate (false,true);

            diagonal[i] = phi.begin_dof_values()[i];
        }

        for (unsigned int i=0; i<phi.dofs_per_cell; ++i)
            phi.begin_dof_values()[i] = diagonal[i];
        phi.distribute_local_to_global (dst);
    }
}




template <int dim>
class LaplaceProblem
{
public:
    LaplaceProblem ();
    void run ();

private:
    void setup_system ();
    void assemble_rhs ();
    void solve ();
    void output_results (const unsigned int cycle) const;

    parallel::distributed::Triangulation<dim>  triangulation;

    FESystem<dim>                                  fe;
    DoFHandler<dim>                            dof_handler;

    ConstraintMatrix                           constraints;
    typedef LaplaceOperator<dim,degree_finite_element,double> SystemMatrixType;
    SystemMatrixType                           system_matrix;

    MGConstrainedDoFs                          mg_constrained_dofs;
    typedef LaplaceOperator<dim,degree_finite_element,float>  LevelMatrixType;
    MGLevelObject<LevelMatrixType>             mg_matrices;

    LinearAlgebra::distributed::Vector<double> solution;
    LinearAlgebra::distributed::Vector<double> system_rhs;

    ConditionalOStream                         pcout;
    ConditionalOStream                         time_details;
};



template <int dim>
LaplaceProblem<dim>::LaplaceProblem ()
    :
      triangulation(MPI_COMM_WORLD,
                    Triangulation<dim>::limit_level_difference_at_vertices,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe (FE_Q<dim>(degree_finite_element), dim),
      dof_handler (triangulation),
      pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      time_details (std::cout, false &&
                    Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
{}





template <int dim>
void LaplaceProblem<dim>::setup_system ()
{
    system_matrix.clear();
    mg_matrices.clear_elements();

    dof_handler.distribute_dofs (fe);
    dof_handler.distribute_mg_dofs ();

    pcout << "Number of degrees of freedom: "
          << dof_handler.n_dofs()
          << std::endl;

    IndexSet locally_relevant_dofs;
    DoFTools::extract_locally_relevant_dofs (dof_handler,
                                             locally_relevant_dofs);

    constraints.clear();
    constraints.reinit(locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            constraints);

    std::set<types::boundary_id> dirichlet_boundary;
    std::set<types::boundary_id> no_flux_boundary;
    // Mixed dirichlet and no-flux
    if (true)
    {
        dirichlet_boundary.insert(0);

        VectorTools::interpolate_boundary_values (dof_handler,
                                                  0,
                                                  Functions::ZeroFunction<dim>(dim),
                                                  constraints);

        no_flux_boundary.insert(1);
        no_flux_boundary.insert(2);
        no_flux_boundary.insert(3);
        VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                        0,
                                                        no_flux_boundary,
                                                        constraints);
    }
    // All no-flux
    else if (false)
    {
        no_flux_boundary.insert(0);
        no_flux_boundary.insert(1);
        no_flux_boundary.insert(2);
        no_flux_boundary.insert(3);
        VectorTools::compute_no_normal_flux_constraints(dof_handler,
                                                        0,
                                                        no_flux_boundary,
                                                        constraints);
    }
    // All Dirichlet
    else
    {
        dirichlet_boundary.insert(0);
        dirichlet_boundary.insert(1);
        dirichlet_boundary.insert(2);
        dirichlet_boundary.insert(3);
        for (unsigned int i=0; i<2*dim; ++i)
            VectorTools::interpolate_boundary_values (dof_handler,
                                                      i,
                                                      Functions::ZeroFunction<dim>(dim),
                                                      constraints);
    }
    constraints.close();

    {
        typename MatrixFree<dim,double>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
                MatrixFree<dim,double>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points);
        std::shared_ptr<MatrixFree<dim,double> >
                system_mf_storage(new MatrixFree<dim,double>());
        system_mf_storage->reinit (dof_handler, constraints, QGauss<1>(fe.degree+1),
                                   additional_data);
        system_matrix.initialize (system_mf_storage);
    }

    system_matrix.evaluate_coefficient(Coefficient<dim>());

    system_matrix.initialize_dof_vector(solution);
    system_matrix.initialize_dof_vector(system_rhs);

    const unsigned int nlevels = triangulation.n_global_levels();
    mg_matrices.resize(0, nlevels-1);

    mg_constrained_dofs.initialize(dof_handler);
    if (dirichlet_boundary.size() > 0)
        mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);

    for (unsigned int level=0; level<nlevels; ++level)
    {
        IndexSet relevant_dofs;
        DoFTools::extract_locally_relevant_level_dofs(dof_handler, level,
                                                      relevant_dofs);
        ConstraintMatrix level_constraints;
        level_constraints.reinit(relevant_dofs);
        level_constraints.add_lines(mg_constrained_dofs.get_boundary_indices(level));
        level_constraints.close();

        typename MatrixFree<dim,float>::AdditionalData additional_data;
        additional_data.tasks_parallel_scheme =
                MatrixFree<dim,float>::AdditionalData::none;
        additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                                update_quadrature_points);
        additional_data.level_mg_handler = level;
        std::shared_ptr<MatrixFree<dim,float> >
                mg_mf_storage_level(new MatrixFree<dim,float>());
        mg_mf_storage_level->reinit(dof_handler, level_constraints,
                                    QGauss<1>(fe.degree+1), additional_data);

        mg_matrices[level].initialize(mg_mf_storage_level, mg_constrained_dofs,
                                      level);
        mg_matrices[level].evaluate_coefficient(Coefficient<dim>());
    }
}




template <int dim>
void LaplaceProblem<dim>::assemble_rhs ()
{
    RightHandSide<dim> rhs_func;
    Tensor<1,dim,VectorizedArray<double> > rhs_val;
    for (unsigned int d=0; d<dim; ++d)
        rhs_val[d] = make_vectorized_array<double> (0.0);

    solution = 0;
    constraints.distribute(solution);
    system_rhs = 0;
    solution.update_ghost_values();
    FEEvaluation<dim,degree_finite_element,degree_finite_element+1,dim,double> phi(*system_matrix.get_matrix_free());

    const Table<2, VectorizedArray<double> > coefficient = system_matrix.get_coefficient_table();
    for (unsigned int cell=0; cell<system_matrix.get_matrix_free()->n_macro_cells(); ++cell)
    {
        phi.reinit(cell);
        phi.read_dof_values_plain(solution);
        phi.evaluate(false, true);
        for (unsigned int q=0; q<phi.n_q_points; ++q)
        {
            phi.submit_gradient(-coefficient(cell,q) * phi.get_gradient(q), q);

            for (unsigned int i=0; i<VectorizedArray<double>::n_array_elements; ++i)
            {
                Point<dim> quad_point;
                for (unsigned int d=0; d<dim; ++d)
                    quad_point(d)=phi.quadrature_point(q)(d)[i];

                Vector<double> rhs_temp(dim);
                rhs_func.vector_value(quad_point,rhs_temp);
                for (unsigned int d=0; d<dim; ++d)
                    rhs_val[d][i] = rhs_temp(d);
            }
            phi.submit_value(rhs_val, q);
        }
        phi.integrate(true, true);
        phi.distribute_local_to_global(system_rhs);
    }
    system_rhs.compress(VectorOperation::add);
}



template <int dim>
void LaplaceProblem<dim>::solve ()
{
    MGTransferMatrixFree<dim,float> mg_transfer(mg_constrained_dofs);
    mg_transfer.build(dof_handler);

    typedef PreconditionChebyshev<LevelMatrixType,LinearAlgebra::distributed::Vector<float> > SmootherType;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<float> >
            mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level = 0; level<triangulation.n_global_levels(); ++level)
    {
        if (level > 0)
        {
            smoother_data[level].smoothing_range = 15.;
            smoother_data[level].degree = 4;
            smoother_data[level].eig_cg_n_iterations = 10;
        }
        else
        {
            smoother_data[0].smoothing_range = 1e-3;
            smoother_data[0].degree = numbers::invalid_unsigned_int;
            smoother_data[0].eig_cg_n_iterations = mg_matrices[0].m();
        }
        mg_matrices[level].compute_diagonal();
        smoother_data[level].preconditioner = mg_matrices[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices, smoother_data);

    MGCoarseGridApplySmoother<LinearAlgebra::distributed::Vector<float> > mg_coarse;
    mg_coarse.initialize(mg_smoother);

    mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_matrix(mg_matrices);

    MGLevelObject<MatrixFreeOperators::MGInterfaceOperator<LevelMatrixType> > mg_interface_matrices;
    mg_interface_matrices.resize(0, triangulation.n_global_levels()-1);
    for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
        mg_interface_matrices[level].initialize(mg_matrices[level]);
    mg::Matrix<LinearAlgebra::distributed::Vector<float> > mg_interface(mg_interface_matrices);

    Multigrid<LinearAlgebra::distributed::Vector<float> > mg(mg_matrix,
                                                             mg_coarse,
                                                             mg_transfer,
                                                             mg_smoother,
                                                             mg_smoother);
    mg.set_edge_matrices(mg_interface, mg_interface);

    PreconditionMG<dim, LinearAlgebra::distributed::Vector<float>,
            MGTransferMatrixFree<dim,float> >
            preconditioner(dof_handler, mg, mg_transfer);


    SolverControl solver_control (100, 1e-8*system_rhs.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double> > cg (solver_control);

    constraints.set_zero(solution);
    cg.solve (system_matrix, solution, system_rhs,
              preconditioner);

    constraints.distribute(solution);

    pcout << "CG solve in "
          << solver_control.last_step()
          << " iterations." << std::endl;
}




template <int dim>
void LaplaceProblem<dim>::output_results (const unsigned int cycle) const
{
    if (triangulation.n_global_active_cells() > 1000000)
        return;

    solution.update_ghost_values();

    std::vector<std::string> solution_names (dim, "velocity");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
            data_component_interpretation
            (dim, DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (solution,
                              solution_names,
                              DataOut<dim>::type_dof_data,
                              data_component_interpretation);
    data_out.build_patches ();

    std::ofstream output ("solution-"
                          + std::to_string(cycle)
                          + "."
                          + std::to_string(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
                          + ".vtu");
    data_out.write_vtu (output);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
        std::vector<std::string> filenames;
        for (unsigned int i=0; i<Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD); ++i)
            filenames.emplace_back("solution-"
                                   + std::to_string(cycle)
                                   + "."
                                   + std::to_string(i)
                                   + ".vtu");

        std::string master_name = "solution-" + Utilities::to_string(cycle) + ".pvtu";
        std::ofstream master_output (master_name.c_str());
        data_out.write_pvtu_record (master_output, filenames);
    }
}




template <int dim>
void LaplaceProblem<dim>::run ()
{
    {
        const unsigned int n_vect_doubles = VectorizedArray<double>::n_array_elements;
        const unsigned int n_vect_bits = 8*sizeof(double)*n_vect_doubles;

        pcout << "Vectorization over " << n_vect_doubles
              << " doubles = " << n_vect_bits << " bits ("
              << Utilities::System::get_current_vectorization_level()
              << "), VECTORIZATION_LEVEL=" << DEAL_II_COMPILER_VECTORIZATION_LEVEL << std::endl;
    }

    for (unsigned int cycle=0; cycle<9-dim; ++cycle)
    {
        pcout << "Cycle " << cycle << std::endl;

        if (cycle == 0)
        {
            GridGenerator::hyper_cube (triangulation, 0., 1., true);
            triangulation.refine_global (3-dim);
        }
        triangulation.refine_global (1);
        setup_system ();
        assemble_rhs ();
        solve ();
        output_results (cycle);
        pcout << std::endl;
    };
}
}




int main (int argc, char *argv[])
{
    try
    {
        using namespace Step37;

        Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

        LaplaceProblem<dimension> laplace_problem;
        laplace_problem.run ();
    }
    catch (std::exception &exc)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception on processing: " << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Unknown exception!" << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        return 1;
    }

    return 0;
}
